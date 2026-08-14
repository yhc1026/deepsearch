"""
RAGFlow 知识库工具模块

封装两个给 RAGFlow 子智能体使用的 LangChain 工具。
所有工具统一返回标准信封 JSON：{"code":"HIT|MISS|ERROR","content":"..."}
"""

import json

from langchain_core.tools import tool
from ragflow_sdk import RAGFlow

from shared.agent_result import ERROR, HIT, MISS, make_result
from shared.monitor import monitor
from agents.ragflow_search.ragflow_config import _load_ragflow_env

api_key, base_url = _load_ragflow_env()
ragflow_client = RAGFlow(api_key=api_key, base_url=base_url)


@tool
def get_assistant_list() -> str:
    """
    查询 RAGFlow 中有哪些聊天助手，以及每个助手关联了哪些知识库

    作用：让模型先了解"哪个助手能回答哪类内部文档问题"，再决定后续要向哪个助手提问。
    调用 create_ask_delete 之前，应先调用本工具确认助手名称。
    :return: 标准信封 JSON（code + content）
    """
    monitor.report_tool(tool_name="ragflow聊天助手列表查询工具：get_assistant_list")

    try:
        chat_list = ragflow_client.list_chats()
        if not chat_list:
            return make_result(MISS, "没有任何可用助手")

        count_chat_info = ""
        for chat in chat_list:
            kb_names = getattr(chat, "kb_names", [])
            if kb_names:
                dataset_names = kb_names
            else:
                dataset_names = []
            count_chat_info += (
                f"助手名称:{chat.name};功能介绍：{chat.description}; "
                f"关联的知识库：{'、'.join(dataset_names)} \n"
            )
        return make_result(HIT, count_chat_info)
    except Exception as e:
        return make_result(ERROR, f"查询助手信息异常: {str(e)}")


@tool
def create_ask_delete(chat_name, question) -> str:
    """
    向某个 RAGFlow 聊天助手创建临时会话并完成一次提问

    注意：调用此工具之前，必须先调用 get_assistant_list，明确可用助手名称和助手能力边界。
    :param chat_name: 助手名称，必须来自 get_assistant_list 返回结果
    :param question: 本次提问的问题
    :return: 标准信封 JSON（code + content）
    """
    monitor.report_tool(
        tool_name="ragflow提问助手工具：create_ask_delete",
        args={"chat_name": chat_name, "question": question},
    )

    try:
        chats = ragflow_client.list_chats(name=chat_name)
        if not chats:
            return make_result(MISS, f"未找到名为「{chat_name}」的助手")

        use_chat = chats[0]
        if len(question) > 120:
            question_suffix = "…"
        else:
            question_suffix = ""
        print(
            f"\033[37m[RAGFlow] 向「{chat_name}」提问: "
            f"{question[:120]}{question_suffix}\033[0m"
        )

        session = use_chat.create_session(name="temp_session_ask")
        response = ragflow_client.post(
            f"/chats/{use_chat.id}/completions",
            {
                "messages": [{"role": "user", "content": question}],
                "stream": True,
                "session_id": session.id,
            },
            stream=True,
        )
        result = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            line = line.removeprefix("data:").strip()
            if line == "[DONE]":
                break
            data = json.loads(line)
            chunk_data = data.get("data")
            if not isinstance(chunk_data, dict):
                continue
            answer = chunk_data.get("answer")
            if answer:
                if answer.startswith(result):
                    result = answer
                elif not result.startswith(answer):
                    result += answer

        use_chat.delete_sessions(ids=[session.id])
        print(f"\033[37m[RAGFlow] 检索结果 ({len(result)} 字符):\n{result}\033[0m")

        if not result or not result.strip():
            return make_result(MISS, "知识库未返回任何内容")

        # 启发式：RAGFlow 常见“不知道”话术 → MISS（仍带原文便于排查）
        miss_hints = ("不知道", "无法回答", "没有找到", "未找到相关", "不足以", "无关")
        lowered = result.strip()
        if any(h in lowered for h in miss_hints) and len(lowered) < 80:
            return make_result(MISS, result)

        return make_result(HIT, result)
    except Exception as e:
        return make_result(ERROR, f"提问失败，错误原因：{str(e)}")
