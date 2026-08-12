"""
A2A Agent 工具模块

为主智能体提供 3 个 LangChain 工具，每个工具封装对远程子智能体服务的 HTTP 调用。
统一返回标准业务信封 JSON：{"code":"HIT|MISS|ERROR","content":"..."}
"""

import asyncio
import json
import urllib.error
import urllib.request
from typing import Optional

from langchain_core.tools import tool

from shared.agent_result import ERROR, HIT, MISS, make_result, parse_result
from shared.monitor import monitor

SUBAGENT_URLS = {
    "network_search": "http://localhost:8001",
    "database_query": "http://localhost:8002",
    # "ragflow": "http://localhost:8003",  # TODO: 取消注释以启用 RAGFlow
    "vector_search": "http://localhost:8004",
}

TIMEOUT_SECONDS = 120


def _call_subagent_sync(service_key: str, query: str, tool_name: str) -> str:
    """同步 HTTP 调用子智能体，始终返回标准信封 JSON 字符串。"""
    url = f"{SUBAGENT_URLS[service_key]}/tasks"
    payload = json.dumps({"query": query}).encode("utf-8")

    print(f"\033[36m  [SubAgent] → {tool_name} ({url})\033[0m")

    monitor.report_tool(
        tool_name=tool_name,
        args={"service": service_key, "query": query},
    )

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if data.get("status") == "completed":
            # 优先使用服务端显式 result_code
            raw_code = data.get("result_code")
            if raw_code:
                explicit = raw_code.upper()
            else:
                explicit = ""
            raw_result_value = data.get("result")
            if raw_result_value:
                raw_result = raw_result_value
            else:
                raw_result = ""
            if explicit in (HIT, MISS, ERROR):
                _, content = parse_result(raw_result)
                # result 本身已是信封时用其 content；否则整段当 content
                if not content and raw_result:
                    code2, content2 = parse_result(raw_result)
                    if content2:
                        content = content2
                    else:
                        content = raw_result
                if content:
                    result_content = content
                elif raw_result:
                    result_content = raw_result
                else:
                    result_content = ""
                return make_result(explicit, result_content)

            code, content = parse_result(raw_result)
            if content:
                result_content = content
            else:
                result_content = "子智能体返回了空结果"
            return make_result(code, result_content)

        error_msg: str = data.get("error", "未知错误")
        agent_name = data.get("agent_name", service_key)
        raw_err_code = data.get("result_code")
        if raw_err_code:
            explicit_err = raw_err_code.upper()
        else:
            explicit_err = ERROR
        if explicit_err in (HIT, MISS, ERROR):
            result_code = explicit_err
        else:
            result_code = ERROR
        return make_result(
            result_code,
            f"子智能体 [{agent_name}] 执行失败: {error_msg}",
        )

    except TimeoutError:
        return make_result(ERROR, f"调用子智能体超时（>{TIMEOUT_SECONDS}s）: {url}")
    except urllib.error.URLError as e:
        return make_result(ERROR, f"无法连接到子智能体服务 {url}: {e.reason}")
    except json.JSONDecodeError:
        return make_result(ERROR, "子智能体服务返回了无效的响应格式")
    except Exception as e:
        return make_result(ERROR, f"调用子智能体时发生异常: {str(e)}")


async def _call_subagent(service_key: str, query: str, tool_name: str) -> str:
    """异步包装：把阻塞 HTTP 放到线程池。"""
    return await asyncio.to_thread(_call_subagent_sync, service_key, query, tool_name)


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
