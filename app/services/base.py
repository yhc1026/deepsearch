"""
A2A (Agent-to-Agent) 服务基类

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

from app.shared.llm import model
from app.utils.logger import setup_logging

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
    status: str  # "completed" | "failed"
    result: Optional[str] = None
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
        skills = [self.skills_dir] if self.skills_dir else None
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
            capabilities.append({
                "name": tool.name,
                "description": tool.description.split("\n")[0] if tool.description else "",
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
            task_id = request.task_id or str(uuid.uuid4())

            try:
                if self.agent is None:
                    return TaskResponse(
                        task_id=task_id,
                        status="failed",
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

                # 提取最终 AI 消息作为结果
                messages = result.get("messages", [])
                final_msg = ""
                for msg in reversed(messages):
                    if hasattr(msg, "content") and msg.content and msg.type == "ai":
                        final_msg = msg.content
                        break

                if not final_msg:
                    final_msg = "Agent 未返回有效结果"

                return TaskResponse(
                    task_id=task_id,
                    status="completed",
                    result=final_msg,
                    agent_name=self.name,
                )

            except asyncio.TimeoutError:
                return TaskResponse(
                    task_id=task_id,
                    status="failed",
                    error=f"任务执行超时（>{TASK_TIMEOUT_SECONDS}s）",
                    agent_name=self.name,
                )
            except Exception as e:
                return TaskResponse(
                    task_id=task_id,
                    status="failed",
                    error=str(e),
                    agent_name=self.name,
                )

        @app.get("/health")
        async def health():
            """健康检查"""
            return {"status": "ok", "agent": self.name}

        return app
