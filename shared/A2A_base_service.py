"""
A2A (Agent-to-Agent) 服务基类
Agent实现最重要的内核，让Agents获得web应用的能力

基于 Google 官方 a2a-sdk 实现标准 A2A 协议（JSON-RPC 2.0）：
- Agent Card 自动发现端点 (GET /.well-known/agent-card.json)
- 同步消息发送 (message/send)，无流式
- 结果信封（result_code）通过 DataPart 承载

子类只需提供 name, description, tools, system_prompt 即可得到一个独立的 Agent 服务。
"""

import asyncio
import os
import uuid
import warnings
from contextlib import asynccontextmanager
from typing import Any, Optional

# 先导入 langchain_core 触发其 surface_langchain_deprecation_warnings（把弃用告警设为 default），
# 再关闭 PendingDeprecationWarning，避免 langgraph 导入时刷屏
import langchain_core
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

from deepagents import create_deep_agent
from fastapi import FastAPI
from langgraph.checkpoint.memory import InMemorySaver

from a2a.helpers.proto_helpers import new_data_part, new_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.fastapi_routes import add_a2a_routes_to_fastapi
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Role,
)
from a2a.utils.constants import PROTOCOL_VERSION_1_0, TransportProtocol

from shared.llm import model
from shared.logger import setup_logging
from shared.agent_result import ERROR, HIT, MISS, parse_result

# 略小于客户端 A2A timeout(120s)，避免服务端无限跑、客户端已放弃
TASK_TIMEOUT_SECONDS = 110
RECURSION_LIMIT = 22


class _A2AAgentExecutor(AgentExecutor):
    """将 DeepAgent 的 ainvoke 适配为官方 A2A AgentExecutor。

    采用「即时响应」工作流：execute 结束后仅入队一个 Message，
    其中 text part 承载可读正文，data part 承载业务信封 {code, content}。
    """

    def __init__(self, service: "A2AAgentService") -> None:
        self._service = service

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        query = context.get_user_input()
        task_id = context.task_id or str(uuid.uuid4())
        context_id = context.context_id

        if self._service.agent is None:
            result_code, content = ERROR, "Agent 尚未初始化完成，请稍后重试"
        else:
            try:
                result = await asyncio.wait_for(
                    self._service.agent.ainvoke(
                        {
                            "messages": [
                                {"role": "user", "content": query}
                            ]
                        },
                        config={
                            "configurable": {"thread_id": task_id},
                            "recursion_limit": RECURSION_LIMIT,
                        },
                    ),
                    timeout=TASK_TIMEOUT_SECONDS,
                )
                result_code, content = _extract_result_from_messages(
                    result.get("messages", [])
                )
            except asyncio.TimeoutError:
                result_code = ERROR
                content = f"任务执行超时（>{TASK_TIMEOUT_SECONDS}s）"
            except Exception as e:  # noqa: BLE001
                result_code = ERROR
                content = str(e)

        message = new_message(
            parts=[
                new_text_part(content),
                new_data_part({"code": result_code, "content": content}),
            ],
            context_id=context_id,
            task_id=task_id,
            role=Role.ROLE_AGENT,
        )
        await event_queue.enqueue_event(message)

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        # 本次改造不接入任务取消
        return


class A2AAgentService:
    """A2A Agent 服务基类（官方 a2a-sdk）

    封装：
    1. FastAPI 应用创建（挂载官方 A2A 路由）
    2. Agent Card 端点 (GET /.well-known/agent-card.json)
    3. JSON-RPC 消息发送端点 (POST /)
    4. 健康检查端点 (GET /health)
    5. DeepAgent 创建与执行

    使用方式:
        service = A2AAgentService(
            name="网络搜索助手",
            description="负责进行网络知识搜索...",
            tools=[internet_search],
            system_prompt="你是一个专业的网络信息查询助手...",
            base_url="http://localhost:8001",
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
        base_url: Optional[str] = None,
    ):
        self.name = name
        self.description = description
        self.tools = tools
        self.system_prompt = system_prompt
        self.skills_dir = skills_dir
        self.base_url = base_url
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

    def _resolve_base_url(self, base_url: str | None) -> str:
        """解析本服务对外广播的基础 URL（Agent Card 自动发现用）。"""
        resolved = base_url or self.base_url or os.getenv("A2A_BASE_URL")
        if not resolved:
            raise ValueError(
                "缺少 base_url：请在 A2AAgentService(base_url=...) 或环境变量 "
                "A2A_BASE_URL 中提供本服务的对外地址。"
            )
        return resolved.rstrip("/")

    def _build_agent_card(self, base_url: str) -> AgentCard:
        """构建官方 A2A Agent Card。"""
        skills = []
        for tool in self.tools:
            description = (tool.description or "").split("\n")[0]
            skills.append(
                AgentSkill(
                    id=f"skill-{tool.name}",
                    name=tool.name,
                    description=description,
                    tags=["tool"],
                )
            )

        return AgentCard(
            name=self.name,
            description=self.description,
            version="1.0.0",
            supported_interfaces=[
                AgentInterface(
                    url=base_url,
                    protocol_binding=TransportProtocol.JSONRPC,
                    protocol_version=PROTOCOL_VERSION_1_0,
                )
            ],
            capabilities=AgentCapabilities(),
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain", "application/json"],
            skills=skills,
        )

    def build_app(self, lifespan=None, base_url: str | None = None) -> FastAPI:
        """构建 FastAPI 应用，注册官方 A2A 标准端点

        lifespan 用于异步初始化（如 MCP 工具加载），在 uvicorn 事件循环中执行。
        """
        resolved_base_url = self._resolve_base_url(base_url)
        agent_card = self._build_agent_card(resolved_base_url)

        executor = _A2AAgentExecutor(self)
        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
            agent_card=agent_card,
        )

        if lifespan is None:

            @asynccontextmanager
            async def _default_lifespan(_app: FastAPI):
                setup_logging()
                yield

            lifespan = _default_lifespan

        @asynccontextmanager
        async def _lifespan(_app: FastAPI):
            async with lifespan(_app):
                yield
            await request_handler.aclose()

        app = FastAPI(title=self.name, lifespan=_lifespan)
        add_a2a_routes_to_fastapi(
            app,
            agent_card_routes=create_agent_card_routes(agent_card),
            jsonrpc_routes=create_jsonrpc_routes(request_handler, rpc_url="/"),
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
