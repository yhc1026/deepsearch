"""
主智能体组装与异步执行模块 (单机多进程版)

负责把模型、主提示词、文件类工具和 A2A 远程专家工具组装成 DeepAgent。
主智能体承担三个核心职责：
1. Query 重写：将用户模糊问题转为自包含的子智能体查询
2. 路由分发：根据问题类型选择合适的 A2A 专家工具
3. 结果汇总：整合各专家返回的信息，生成最终交付物

三个专家子智能体已拆分为独立进程（A2A 服务）：
- port 8001: 网络搜索服务
- port 8002: 数据库查询服务
- port 8003: RAGFlow 知识库服务
"""

import asyncio
import shutil
from pathlib import Path

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.llm import model
from app.agent.prompts import main_agent_content
from app.api.context import (
    reset_session_context,
    set_session_context,
    set_thread_context,
)
from app.api.monitor import monitor

# 文件类工具：主智能体直接掌握，负责读取上传附件和生成最终交付文档
from app.tools.markdown_tools import generate_markdown
from app.tools.pdf_tools import convert_md_to_pdf
from app.tools.upload_file_read_tool import read_file_content

# A2A 远程专家工具：替代原来的 DeepAgents 字典式子智能体
# 每个工具通过 HTTP 调用对应端口的独立 Agent 服务
from app.tools.a2a_agent_tools import (
    call_database_query,
    call_network_search,
    call_ragflow_query,
)

# 主智能体: tools 包含 3 个本地文件工具 + 3 个 A2A 远程专家工具
# skills 从 prompts.yml 的 main_agent.skills_dir 加载自定义技能
# subagents 参数不再使用，专家能力通过 A2A 工具的 docstring 暴露给模型
_skills_dir = main_agent_content.get("skills_dir")
main_agent = create_deep_agent(
    model=model,
    system_prompt=main_agent_content["system_prompt"],
    tools=[
        generate_markdown,
        convert_md_to_pdf,
        read_file_content,
        call_network_search,
        call_database_query,
        call_ragflow_query,
    ],
    skills=[_skills_dir] if _skills_dir else None,
    checkpointer=InMemorySaver(),
)

# 当前文件位于 app/agent/main_agent.py，parents[1] 即 app 目录
project_root_path = Path(__file__).parents[1].resolve()


async def run_deep_agent(task_query, session_id):
    """
    异步流式执行主智能体

    API 层会为每次任务传入用户问题和 session_id。本函数负责准备会话目录、
    复制上传文件、写入 ContextVar，并在流式执行过程中把关键事件上报给前端。
    :param task_query: 前端提交的原始任务问题
    :param session_id: 当前任务 ID，同时用于 thread_id、输出目录和 WebSocket 定向推送
    """
    print(f"[MainAgent] 开始执行会话，session_id={session_id}")

    # 每个会话独立使用 output/session_{session_id}，避免不同用户的产物互相覆盖
    session_dir = project_root_path / "output" / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    # 前端和工具使用绝对路径；提示词里只给模型相对路径，降低模型误用系统绝对路径的概率
    session_dir_str = str(session_dir).replace("\\", "/")
    relative_session_dir_str = str(session_dir.relative_to(project_root_path)).replace(
        "\\", "/"
    )

    # 上传文件先落在 updated/session_{session_id}，执行前复制到本次 output 工作目录
    # 这样读文件工具和生成文件工具都只需要围绕同一个 session_dir 工作
    updated_dir_path = project_root_path / "updated" / f"session_{session_id}"
    updated_info_prompt = ""
    if updated_dir_path.exists():
        files = [f.name for f in updated_dir_path.iterdir() if f.is_file()]
        if files:
            for filename in files:
                # copy2 会保留上传文件的修改时间、权限等元数据，便于后续排查文件来源
                shutil.copy2(updated_dir_path / filename, session_dir / filename)

            # 把上传文件列表注入用户消息，提醒模型先调用 read_file_content 获取附件内容
            updated_info_prompt = (
                "\n    [已上传文件] 已加载到工作目录:\n"
                + "\n".join([f"    - {f}" for f in files])
                + "\n    请优先使用工具（read_file_content）读取并参考这些文件。"
            )

    # ContextVar 让深层工具无需显式传参，也能拿到当前会话目录和 WebSocket thread_id
    session_dir_token = set_session_context(session_dir_str)
    session_id_token = set_thread_context(session_id)

    # 前端拿到工作目录后，可以展示本次任务生成的 Markdown/PDF 等产物
    monitor.report_session_dir(session_dir_str)

    # checkpointer 依赖 thread_id 区分会话记忆；同一 session_id 会复用同一条执行上下文
    config = {"configurable": {"thread_id": session_id}}

    # 工作环境指令是运行时动态补充的，约束模型只在当前会话目录读写文件
    path_instruction = f"""
    【工作环境指令】
    工作目录: {relative_session_dir_str}
    {updated_info_prompt}

    规则：
    1. 新生成文件必须保存到工作目录：'{relative_session_dir_str}/filename'
    2. 读取已上传的文件时，请直接将文件名（例如：'开篇.txt'）作为 filename 参数传入（read_file_content）读取工具，不要带上任何目录前缀。
    3. 使用相对路径，禁止使用绝对路径
    4. 若存在上传文件，请先分析内容
    """

    try:
        final_ai_content = ""

        # astream 会持续产出模型节点和工具节点的状态片段
        async for chunk in main_agent.astream(
            {"messages": [{"role": "user", "content": task_query + path_instruction}]},
            config=config,
        ):
            for node_name, state in chunk.items():
                if not state or "messages" not in state:
                    continue
                messages = state["messages"]
                if messages and isinstance(messages, list):
                    last_msg = messages[-1]
                    if node_name == "model":
                        if last_msg.tool_calls:
                            # 模型决定调用工具时，立即推送到前端，让用户看到进度
                            for tool_call in last_msg.tool_calls:
                                tool_name = tool_call["name"]
                                tool_args = tool_call.get("args", {})
                                tool_query = tool_args.get("query", "")[:100]
                                print(
                                    f"[MainAgent] 调用工具: {tool_name}, "
                                    f"query={tool_query}"
                                )
                                monitor.report_tool(
                                    tool_name=tool_name,
                                    args=tool_args,
                                )
                        elif last_msg.content:
                            # 模型产出了思考或规划文本（非工具调用），也推送给前端
                            content_preview = last_msg.content[:200]
                            print(
                                f"[MainAgent] 模型输出: {content_preview}..."
                            )
                            final_ai_content = last_msg.content
                            monitor._emit(
                                "agent_thinking",
                                f"智能体思考中: {content_preview}",
                                {"content": last_msg.content},
                            )
                    elif node_name == "tools":
                        # 工具执行完成，告知前端工具已返回结果
                        if hasattr(last_msg, "name") and hasattr(last_msg, "content"):
                            result_len = len(str(last_msg.content))
                            monitor._emit(
                                "tool_result",
                                f"工具 {last_msg.name} 执行完成，返回 {result_len} 字符",
                                {
                                    "tool_name": last_msg.name,
                                    "result_length": result_len,
                                },
                            )

        # 流式完成后报告最终结果
        if final_ai_content:
            monitor.report_task_result(final_ai_content)

    except asyncio.CancelledError:
        monitor.report_task_cancelled()
        raise
    except Exception as e:
        # 异步执行异常也走 monitor，保证前端能收到明确错误事件
        monitor._emit("error", f"执行主智能发生异常信息：{str(e)}")
    finally:
        # 任务结束后恢复 ContextVar，避免后续请求复用到本次会话目录或 thread_id
        reset_session_context(session_dir_token, session_id_token)


if __name__ == "__main__":
    import asyncio

    asyncio.run(
        run_deep_agent("从网络查询机器人信息，并生成Markdown文件", "test_session_001")
    )
