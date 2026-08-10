"""
A2A (Agent-to-Agent) 服务基类
Agent实现最重要的内核，让Agents获得web应用的能力

封装 FastAPI 应用创建、Agent Card 端点、任务执行端点、健康检查端点，
以及 DeepAgent 的创建与执行。子类只需提供 name, description, tools,
system_prompt 即可得到一个独立的 Agent 服务。
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from deepagents import create_deep_agent
from fastapi import FastAPI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from shared.llm import model
from shared.logger import setup_logging
from shared.agent_result import ERROR, HIT, MISS, parse_result

# 略小于客户端 A2A timeout(120s)，避免服务端无限跑、客户端已放弃
TASK_TIMEOUT_SECONDS = 110
RECURSION_LIMIT = 50


class TaskRequest(BaseModel):
    """A2A 任务请求体"""

    task_id: Optional[str] = None
    query: str


class TaskResponse(BaseModel):
    """A2A 任务响应体"""

    task_id: str
    status: str  # "completed" | "failed"  —— 传输层
    result: Optional[str] = None
    result_code: Optional[str] = None  # HIT | MISS | ERROR —— 业务层
    error: Optional[str] = None
    agent_name: str


class A2AAgentService:
    """A2A Agent 服务基类

    封装：
    1. FastAPI 应用创建
    2. Agent Card 端点 (GET /)
    3. 任务执行端点 (POST /tasks)
    4. 健康检查端点 (GET /health)
    5. DeepAgent 创建与执行

    使用方式:
        service = A2AAgentService(
            name="网络搜索助手",
            description="负责进行网络知识搜索...",
            tools=[internet_search],
            system_prompt="你是一个专业的网络信息查询助手...",
        )
        service.create_agent()
        app = service.build_app()
    """

    def __init__(
        self,
        name: str,
        description: str,
        tools: list,
        system_prompt: str,
        skills_dir: Optional[str] = None,
    ):
        self.name = name
        self.description = description
        self.tools = tools
        self.system_prompt = system_prompt
        self.skills_dir = skills_dir
        self.agent: Any = None
        self.checkpointer: Any = None

    def create_agent(self) -> None:
        """创建本服务的 DeepAgent 实例。

        首次调用时创建新的 InMemorySaver；后续调用 recreate_agent() 时复用。
        """
        if self.skills_dir:
            skills = [self.skills_dir]
        else:
            skills = None
        if self.checkpointer is None:
            self.checkpointer = InMemorySaver()
        self.agent = create_deep_agent(
            model=model,
            system_prompt=self.system_prompt,
            tools=self.tools,
            skills=skills,
            checkpointer=self.checkpointer,
        )

    def recreate_agent(self, tools: list | None = None, system_prompt: str | None = None) -> None:
        """用新的 tools / system_prompt 重建 agent，复用已有 checkpointer。

        复用 checkpointer 保证重建前的对话记忆不会丢失。
        """
        if tools is not None:
            self.tools = tools
        if system_prompt is not None:
            self.system_prompt = system_prompt
        self.create_agent()

    def _build_capabilities(self) -> list[dict]:
        """从 tools 列表提取能力描述"""
        capabilities = []
        for tool in self.tools:
            if tool.description:
                description = tool.description.split("\n")[0]
            else:
                description = ""
            capabilities.append({
                "name": tool.name,
                "description": description,
            })
        return capabilities

    def build_app(self, lifespan=None) -> FastAPI:
        """构建 FastAPI 应用，注册 A2A 标准端点

        lifespan 用于异步初始化（如 MCP 工具加载），在 uvicorn 事件循环中执行。
        """
        if lifespan is None:

            @asynccontextmanager
            async def _default_lifespan(_app: FastAPI):
                setup_logging()
                yield

            lifespan = _default_lifespan

        app = FastAPI(title=self.name, lifespan=lifespan)

        @app.get("/")
        async def agent_card():
            """Agent Card: 返回本服务的身份和能力描述"""
            return {
                "name": self.name,
                "description": self.description,
                "version": "1.0.0",
                "capabilities": self._build_capabilities(),
                "endpoints": {
                    "tasks": "/tasks",
                    "health": "/health",
                },
            }

        @app.post("/tasks", response_model=TaskResponse)
        async def execute_task(request: TaskRequest):
            """执行 A2A 任务: 接收查询 → 运行 Agent → 返回结果"""
            if request.task_id:
                task_id = request.task_id
            else:
                task_id = str(uuid.uuid4())

            try:
                if self.agent is None:
                    return TaskResponse(
                        task_id=task_id,
                        status="failed",
                        result_code=ERROR,
                        error="Agent 尚未初始化完成，请稍后重试",
                        agent_name=self.name,
                    )

                result = await asyncio.wait_for(
                    self.agent.ainvoke(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": request.query,
                                }
                            ]
                        },
                        config={
                            "configurable": {"thread_id": task_id},
                            "recursion_limit": RECURSION_LIMIT,
                        },
                    ),
                    timeout=TASK_TIMEOUT_SECONDS,
                )

                result_code, final_msg = _extract_result_from_messages(
                    result.get("messages", [])
                )

                return TaskResponse(
                    task_id=task_id,
                    status="completed",
                    result=final_msg,
                    result_code=result_code,
                    agent_name=self.name,
                )

            except asyncio.TimeoutError:
                return TaskResponse(
                    task_id=task_id,
                    status="failed",
                    result_code=ERROR,
                    error=f"任务执行超时（>{TASK_TIMEOUT_SECONDS}s）",
                    agent_name=self.name,
                )
            except Exception as e:
                return TaskResponse(
                    task_id=task_id,
                    status="failed",
                    result_code=ERROR,
                    error=str(e),
                    agent_name=self.name,
                )

        @app.get("/health")
        async def health():
            """健康检查"""
            return {"status": "ok", "agent": self.name}

        return app


def _extract_result_from_messages(messages: list) -> tuple[str, str]:
    """从 Agent 消息链提取 (result_code, content)。

    优先级：
    1. 最后一次工具返回的标准信封（代码侧产出，最可靠）
    2. 最终 AI 消息中的信封
    3. 最终 AI 纯文本 → 默认 HIT
    """
    last_tool_code: Optional[str] = None
    tool_contents: list[str] = []

    for msg in messages:
        msg_type = getattr(msg, "type", None)
        content = getattr(msg, "content", None)
        if not content:
            continue
        if msg_type == "tool":
            code, body = parse_result(content)
            last_tool_code = code
            if body:
                tool_contents.append(body)

    final_ai = ""
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai" and getattr(msg, "content", None):
            final_ai = str(msg.content)
            break

    if final_ai:
        ai_code, ai_content = parse_result(final_ai)
    else:
        ai_code, ai_content = None, ""

    if last_tool_code is not None:
        if ai_content:
            content = ai_content
        elif tool_contents:
            content = "\n\n".join(tool_contents)
        else:
            content = final_ai
        if not content:
            content = "Agent 未返回有效结果"
            if last_tool_code == HIT:
                return MISS, content
        return last_tool_code, content

    if ai_code:
        if ai_content:
            message_content = ai_content
        elif final_ai:
            message_content = final_ai
        else:
            message_content = "Agent 未返回有效结果"
        return ai_code, message_content

    if final_ai:
        return HIT, final_ai
    return MISS, "Agent 未返回有效结果"
