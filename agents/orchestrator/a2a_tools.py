"""
A2A Agent 工具模块（官方 a2a-sdk）

为主智能体提供 3 个 LangChain 工具，每个工具封装对远程子智能体服务的调用。
通过 A2ACardResolver 自动发现子智能体，发送同步 message/send 请求，
从返回 Message 的 DataPart 中重建标准业务信封 JSON：{"code":"HIT|MISS|ERROR","content":"..."}
"""

from a2a.client import ClientCallContext, ClientConfig, ClientFactory
from a2a.helpers.proto_helpers import (
    get_data_parts,
    get_text_parts,
    new_text_message,
)
from a2a.types import Role, SendMessageRequest
from langchain_core.tools import tool

from shared.agent_result import ERROR, make_result, parse_result
from shared.monitor import monitor

SUBAGENT_URLS = {
    "network_search": "http://localhost:8001",
    "database_query": "http://localhost:8002",
    # "ragflow": "http://localhost:8003",  # TODO: 取消注释以启用 RAGFlow
    "vector_search": "http://localhost:8004",
    "memory_agent": "http://localhost:8005",
}

TIMEOUT_SECONDS = 120


async def _call_subagent(service_key: str, query: str, tool_name: str) -> str:
    """通过官方 a2a-sdk 调用子智能体，始终返回标准信封 JSON 字符串。"""
    base_url = SUBAGENT_URLS[service_key]

    print(f"\033[37m  [SubAgent] → {tool_name} ({base_url})\033[0m")

    monitor.report_tool(
        tool_name=tool_name,
        args={"service": service_key, "query": query},
    )

    client = None
    try:
        factory = ClientFactory(ClientConfig(streaming=False))
        client = await factory.create_from_url(
            base_url,
            resolver_http_kwargs={"timeout": TIMEOUT_SECONDS},
        )

        request = SendMessageRequest(
            message=new_text_message(query, role=Role.ROLE_USER)
        )

        async for stream_response in client.send_message(
            request,
            context=ClientCallContext(timeout=TIMEOUT_SECONDS),
        ):
            if not stream_response.HasField("message"):
                continue

            parts = stream_response.message.parts
            data_parts = get_data_parts(parts)
            if data_parts:
                code, content = parse_result(data_parts[0])
            else:
                text = "\n".join(get_text_parts(parts))
                code, content = parse_result(text)
            return make_result(code, content)

        return make_result(ERROR, "子智能体返回了空结果")

    except Exception as e:  # noqa: BLE001
        return make_result(ERROR, f"调用子智能体 [{service_key}] 失败: {str(e)}")
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception:  # noqa: BLE001
                pass


@tool
async def call_network_search(query: str) -> str:
    """调用网络搜索助手，从互联网检索公开信息。

    重要：传入的 query 必须是自包含的完整搜索语句。你需要先对用户的原始问题进行重写:
    - 将相对时间转为绝对日期（如"昨天"→具体日期，"最近"→日期范围）
    - 补全搜索维度（如"天气"→"气温 降水 湿度 风速 天气状况"）
    - 消解指代（如"竞品"→具体公司/产品名称）

    适用场景：新闻、政策、行业趋势、天气、百科知识、网页资料等公开信息。
    """
    return await _call_subagent("network_search", query, "A2A-网络搜索助手")


@tool
async def call_database_query(query: str) -> str:
    """调用数据库查询助手，查询企业结构化业务数据。

    重要：传入的 query 必须是自包含的完整查询描述。你需要先对用户的原始问题进行重写:
    - 将相对时间转为绝对日期或日期范围
    - 描述需要查询的数据维度（如"药品名称、批号、库存数量、仓库位置、效期"）
    - 说明筛选条件和聚合方式
    - 不要写 SQL 语句，用自然语言描述数据需求即可

    适用场景：药品信息、库存、销售记录等企业内部结构化数据查询。
    """
    return await _call_subagent("database_query", query, "A2A-数据库查询助手")


# @tool
# async def call_ragflow_query(query: str) -> str:
#     """调用RAGFlow助手，查询企业内部私有知识库中的非结构化文档。
#
#     重要：传入的 query 必须是自包含的完整问题。你需要先对用户的原始问题进行重写:
#     - 明确要检索的知识领域和主题
#     - 消解指代和简称
#     - 复杂问题分解为多角度查询
#
#     适用场景：PDF、白皮书、研报、制度文件、产品资料等内部文档的知识检索。
#     """
#     return await _call_subagent("ragflow", query, "A2A-RAGFlow助手")
# TODO: 取消注释以上代码块以启用 RAGFlow

@tool
async def call_vector_search(query: str, collection: str = "default") -> str:
    """调用向量检索助手，在内部向量知识库中执行混合检索（语义向量 + 关键词）。

    重要：传入的 query 必须是自包含的完整检索问题。你需要先对用户的原始问题进行重写:
    - 明确要检索的知识领域和主题
    - 消解指代和简称
    - 包含完整的关键词

    检索能力：支持多路召回（3 query 变体）、向量语义检索 + 关键词匹配的混合策略。
    适用场景：企业内部非结构化文档的知识检索。
    """
    return await _call_subagent("vector_search", query, "A2A-向量检索助手")


async def call_memory_agent(message: str) -> str:
    """向长期记忆 Agent 发送记忆提取请求（由 orchestrator 结尾 fire-and-forget 调用）。"""
    return await _call_subagent("memory_agent", message, "A2A-记忆助手")
