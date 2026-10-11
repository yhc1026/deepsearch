"""
主智能体组装与异步执行模块 (Plan-then-Execute DAG 编排版)

核心流程由原来的 ReAct Tool-Calling 循环改为三阶段流水线：

Phase 1 - 规划 (Planner):   LLM 分析用户请求 → 产出结构化 DAG 执行计划
Phase 2 - 执行 (Executor):  确定性代码按拓扑顺序分批执行计划中的每一步
Phase 3 - 汇总 (Synthesize): LLM 整合各步骤结果 → 生成最终答案或交付文档

三个专家子智能体已拆分为独立进程（A2A 服务）：
- port 8001: 网络搜索服务  (call_network_search)
- port 8002: 数据库查询服务 (call_database_query)
- port 8003: RAGFlow 知识库  (call_ragflow_query)

关键保证：
- 计划的 depends_on 字段声明步骤依赖关系
- 拓扑排序保证 B 一定在 A 之后执行
- 代码遍历整个步骤列表，不会跳步或遗漏
- 无依赖步骤自动并行执行 (asyncio.gather)
"""

import asyncio
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from shared.llm import model
from shared.token_usage import add_usage, begin_request, get_usage, reset_request

logger = logging.getLogger(__name__)
from agents.orchestrator.planner import Planner, Plan
from agents.orchestrator.executor import DAGExecutor
from shared.prompts import main_agent_content
from agents.orchestrator.context import (
    reset_session_context,
    set_session_context,
    set_thread_context,
)
from agents.orchestrator.session_db import (
    finish_turn,
    get_context_summary,
    get_conversations,
    save_conversation,
    upsert_session,
)
from agents.orchestrator.checkpoint import CheckpointStore
from agents.orchestrator.memory_client import recall_memories, schedule_memory_extraction
from shared.agent_result import HIT
from shared.monitor import monitor

project_root_path = Path(__file__).parents[2].resolve()

# 工具名简写，打印更短
_TOOL_ABBREV = {
    "call_network_search":  "网络搜索",
    "call_database_query":  "数据库",
    # "call_ragflow_query":   "RAGFlow",  # TODO: 取消注释以启用 RAGFlow
    "call_vector_search":   "向量检索",
    "call_document_writer": "文档撰写",
    "read_file_content":    "读附件",
}

# 向量检索置信度阈值：rerank 分数超过该值即认为答案在内部库，对网络结果做取舍
_VECTOR_AUTHORITATIVE_THRESHOLD = 0.75  # 极高 → 舍弃网络结果
_VECTOR_PREFERRED_THRESHOLD = 0.5       # 较高 → 降权网络结果


def _print_plan(plan: Plan) -> None:
    """在控制台打印简化的执行计划面板，方便人工校验。"""
    bar = "=" * 64
    print(f"\n{bar}")
    print(f"  Plan: {plan.goal}")
    print(f"  Steps: {len(plan.steps)}")
    print(bar)

    batches = _topo_label(plan)
    for batch_idx, batch in enumerate(batches):
        for step in batch:
            tag = _TOOL_ABBREV.get(step.tool, step.tool)
            deps = ""
            if step.depends_on:
                deps = f"  ← 依赖: {', '.join(step.depends_on)}"
            # 截断 query 前 80 字符
            preview = step.query.replace("\n", " ")[:80]
            print(f"  [{tag}] {preview}{deps}")
        # 批次间加空行分隔
        if batch_idx < len(batches) - 1:
            gutter = "  |"
            for i, s in enumerate(batch):
                gutter += f" {s.id} 等结果 →"
            print(f"{gutter}\n")

    print(f"{bar}\n")


def _topo_label(plan: Plan) -> list[list]:
    """简单拓扑分层，按 depends_on 分组，与 executor 的拓扑排序逻辑一致。"""
    step_map = {s.id: s for s in plan.steps}
    in_degree = {s.id: len(s.depends_on) for s in plan.steps}
    dependents: dict[str, list] = {s.id: [] for s in plan.steps}
    for s in plan.steps:
        for dep in s.depends_on:
            dependents[dep].append(s.id)

    batches = []
    current = [s for s in plan.steps if in_degree[s.id] == 0]
    while current:
        batches.append(current)
        nxt = []
        for s in current:
            for did in dependents[s.id]:
                in_degree[did] -= 1
                if in_degree[did] == 0:
                    nxt.append(step_map[did])
        current = nxt
    return batches

# =============================================================================
# 主入口
# =============================================================================


async def run_deep_agent(task_query: str, session_id: str, user_id: int | None = None):
    """三阶段流水线执行主智能体任务。

    Phase 1: 规划 → Planner 将用户请求转换为结构化 DAG 计划
    Phase 2: 执行 → Executor 按拓扑顺序分批调度子 Agent
    Phase 3: 汇总 → LLM 整合结果生成最终交付物

    结束后 fire-and-forget 调度一次长期记忆提取（按 user_id 隔离）。
    """
    logger.info("主智能体开始进行回复")

    # ---- 初始化本次请求的 token 累计器 ----
    begin_request()

    # ---- 准备会话工作目录 ----
    session_dir = project_root_path / "output" / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    session_dir_str = str(session_dir).replace("\\", "/")
    relative_session_dir_str = str(
        session_dir.relative_to(project_root_path)
    ).replace("\\", "/")

    # ---- 复制上传文件到工作目录 ----
    uploaded_context = ""
    updated_dir_path = project_root_path / "updated" / f"session_{session_id}"
    if updated_dir_path.exists():
        files = [f.name for f in updated_dir_path.iterdir() if f.is_file()]
        if files:
            for filename in files:
                shutil.copy2(updated_dir_path / filename, session_dir / filename)
            uploaded_context = (
                "\n[已上传文件]\n"
                + "\n".join(f"  - {f}" for f in files)
                + "\n请优先关注这些文件中的内容。"
            )

    # ---- ContextVar 注入，深层工具通过 get_session_context 获取 ----
    session_dir_token = set_session_context(session_dir_str)
    session_id_token = set_thread_context(session_id)
    monitor.set_current_thread_id(session_id)
    monitor.set_current_session_dir(session_dir_str)
    monitor.report_session_dir(session_dir_str)

    # ---- 当前日期上下文（供 Planner 将相对时间转为绝对日期） ----
    today = datetime.now()
    date_context = (
        f"\n[当前日期] {today.strftime('%Y年%m月%d日')} (星期{['一','二','三','四','五','六','日'][today.weekday()]})"
        f"\n[工作目录] {relative_session_dir_str}"
        f"{uploaded_context}"
    )

    # ---- 加载历史对话上下文（全局摘要 + 最近3轮完整 + 旧轮次用摘要） ----
    history_context = ""
    try:
        history_turns = get_conversations(session_id)
        if history_turns:
            ctx_summary = get_context_summary(session_id)
            lines = ["\n[历史对话]"]

            if ctx_summary:
                lines.append(f"[对话概要] {ctx_summary}")

            recent_count = 3
            for i, turn in enumerate(history_turns):
                is_recent = i >= len(history_turns) - recent_count
                lines.append(f"用户: {turn['user_query']}")
                if is_recent:
                    if turn.get("assistant_result"):
                        lines.append(f"助手: {turn['assistant_result']}")
                else:
                    content = turn.get("summary") or turn.get("assistant_result", "")
                    if content:
                        lines.append(f"助手: {content[:300]}")

            history_context = "\n" + "\n".join(lines)
            if len(history_context) > 6000:
                history_context = history_context[:6000] + "\n...(历史对话已截断)"
    except Exception:
        logger.warning("加载历史上下文失败", exc_info=True)

    # ---- 长期记忆召回：硬编码强制，同步调用，命中则注入主智能体 prompt ----
    recall_context = ""
    if user_id is not None:
        try:
            recall_code, recall_content = await recall_memories(
                user_id, session_id, task_query
            )
            if recall_code == HIT and recall_content:
                recall_context = (
                    "\n[用户长期记忆]\n"
                    + recall_content
                    + "\n以上是该用户相关的历史记忆，仅作背景参考；"
                    + "仍需规划信息检索步骤，获取最新、准确的数据后作答。"
                )
        except Exception:
            logger.warning("长期记忆召回失败", exc_info=True)

    full_query = task_query + date_context + history_context + recall_context
    last_result: str = ""
    session_files: list[dict] = []
    checkpoint_store = CheckpointStore()
    turn_index: int = 0

    try:
        # ================================================================
        # Phase 1: 规划
        # ================================================================
        print("\033[37m▶ Phase 1/3: 规划 — 分析用户意图，生成 DAG 执行计划\033[0m")

        # 同一 session + 同一 query 存在未完成断点时，跳过规划直接恢复原计划
        resume = await checkpoint_store.load(session_id, task_query)
        resume_state: dict[str, Any] | None = None
        if resume is not None:
            plan: Plan = resume["plan"]
            resume_state = {
                "results": resume["results"],
                "result_codes": resume["result_codes"],
                "batch_index": resume["batch_index"],
            }
            print(
                f"\033[33m  ↺ 检测到断点，从第 {resume['batch_index'] + 2} 批恢复执行\033[0m"
            )
        else:
            planner = Planner()
            plan = await planner.plan(full_query)

        _print_plan(plan)

        monitor._emit(
            "plan_generated",
            f"执行计划已生成：{plan.goal}（共 {len(plan.steps)} 步）",
            {
                "goal": plan.goal,
                "steps": [
                    {
                        "id": s.id,
                        "tool": s.tool,
                        "description": s.description,
                        "depends_on": s.depends_on,
                    }
                    for s in plan.steps
                ],
            },
        )

        # 空计划 = LLM 判定不需要工具，直接口头回复
        # 注意：必须用 full_query（含记忆召回/历史上下文），否则记忆命中时答案会被丢弃
        if not plan.steps:
            await checkpoint_store.clear(session_id)
            last_result = await _chat_reply(full_query)
            if last_result:
                monitor.report_task_result(last_result)
            return last_result

        # 全新规划且非空计划时落初始断点（batch_index=-1），保证首个 batch 崩溃也能恢复
        if resume is None:
            await checkpoint_store.save(session_id, task_query, plan, {}, {}, -1)

        # ================================================================
        # Phase 2: 执行
        # ================================================================
        print("\033[37m▶ Phase 2/3: 执行 — 按 DAG 拓扑顺序调用子 Agent\033[0m")
        executor = DAGExecutor()
        results, result_scores = await executor.execute(
            plan,
            resume_state=resume_state,
            on_batch_done=lambda results, codes, idx: checkpoint_store.save(
                session_id, task_query, plan, results, codes, idx
            ),
        )

        # 执行完成即清理断点：Phase 3 汇总失败无需重跑检索步骤
        await checkpoint_store.clear(session_id)

        # ================================================================
        # Phase 3: 汇总
        # ================================================================
        print("\033[37m▶ Phase 3/3: 汇总 — 整合各子 Agent 结果，生成最终答案\033[0m")
        last_result = await _synthesize(task_query, plan, results, result_scores)

        if last_result:
            monitor.report_task_result(last_result)

    except asyncio.CancelledError:
        last_result = "任务已取消"
        monitor.report_task_cancelled()
        raise
    except Exception as e:
        last_result = f"执行异常：{e}"
        monitor._emit("error", f"主智能体执行异常：{str(e)}")
        raise
    finally:
        # ---- token 用量上报（主智能体 + 已返回的子智能体） ----
        usage = get_usage()
        if usage["total_tokens"] > 0:
            monitor._emit(
                "usage",
                f"本次请求 token 消耗：输入 {usage['input_tokens']} / "
                f"输出 {usage['output_tokens']} / 总计 {usage['total_tokens']}",
                usage,
            )

        # ---- 收集会话产出文件 ----
        try:
            if session_dir.exists():
                session_files = [
                    {"name": f.name, "path": str(f), "size": f.stat().st_size}
                    for f in session_dir.iterdir()
                    if f.is_file()
                ]
        except Exception:
            pass

        # ---- 持久化对话记录 ----
        if last_result:
            try:
                upsert_session(session_id, task_query[:50], user_id)
                turn_index = save_conversation(
                    session_id, task_query, last_result, session_files, user_id
                )
                finish_turn(session_id)
            except Exception:
                logger.warning("保存对话记录失败", exc_info=True)

        # ---- 长期记忆提取：硬编码强制，末尾异步 fire-and-forget ----
        schedule_memory_extraction(
            user_id, session_id, turn_index, task_query, last_result
        )

        reset_session_context(session_dir_token, session_id_token)
        reset_request()

    return last_result


# =============================================================================
# 闲聊回复（空计划时直接口头回复，不调用子 Agent）
# =============================================================================


async def _chat_reply(task_query: str) -> str:
    """当 Planner 判定不需要工具调用时，直接用 LLM 生成对话回复。"""
    system_prompt = (
        "你是 DeepSearch，一个 AI 深度研搜助手。你可以帮助用户进行行业分析、"
        "市场研究、数据库查询、文档生成等任务。对于简单的问候或闲聊，"
        "请友好简洁地回复，并简要介绍你能做什么。"
    )
    full: list[str] = []
    last_usage = None
    async for chunk in model.astream([
        SystemMessage(content=system_prompt),
        HumanMessage(content=task_query),
    ]):
        if chunk.content:
            full.append(str(chunk.content))
            monitor.stream_chunk(str(chunk.content))
        if chunk.usage_metadata is not None:
            last_usage = chunk.usage_metadata
    monitor.stream_done()
    add_usage(last_usage)
    if full:
        return "".join(full)
    else:
        return "你好，有什么可以帮你的？"


# =============================================================================
# Phase 3: 汇总器
# =============================================================================


async def _synthesize(
    task_query: str,
    plan: Plan,
    results: dict[str, str],
    result_scores: dict[str, float | None] | None = None,
) -> str:
    """汇总所有步骤结果，生成最终交付物。"""
    # 检查是否有文档撰写步骤已完成
    file_steps_done = [
        s.id for s in plan.steps
        if s.tool == "call_document_writer"
    ]

    if file_steps_done:
        # 文件已生成，返回简要摘要
        completed_files = [
            results.get(sid, "").strip()
            for sid in file_steps_done
            if sid in results
        ]
        return (
            f"任务「{plan.goal}」已完成。\n\n"
            + "\n".join(completed_files)
        )

    # 纯问答场景：用 LLM 整合所有信息获取结果
    info_steps = [
        s for s in plan.steps
        if s.tool in ("call_network_search", "call_database_query", "call_vector_search")  # call_ragflow_query 已禁用
    ]

    if not info_steps:
        # 没有信息获取步骤，直接返回第一个结果
        if plan.steps:
            return results.get(plan.steps[0].id, "任务已完成")
        else:
            return "任务已完成"

    # 构造汇总 prompt
    # 向量检索置信度：极高时舍弃网络结果，较高时降权网络结果
    result_scores = result_scores or {}
    vector_score = max(
        (
            result_scores.get(s.id) or 0.0
            for s in info_steps
            if s.tool == "call_vector_search"
        ),
        default=0.0,
    )

    collected_parts: list[str] = []
    authority_note = ""
    for s in info_steps:
        if (
            s.tool == "call_network_search"
            and vector_score >= _VECTOR_AUTHORITATIVE_THRESHOLD
        ):
            # 内部向量库命中置信度极高，网络结果视为噪声舍弃
            continue
        collected_parts.append(
            f"### {s.description} (来源: {s.id})\n{results.get(s.id, '(无结果)')}"
        )

    if vector_score >= _VECTOR_AUTHORITATIVE_THRESHOLD:
        authority_note = "\n\n[来源取舍] 内部向量知识库命中置信度极高，已舍弃网络搜索结果。"
    elif vector_score >= _VECTOR_PREFERRED_THRESHOLD:
        authority_note = "\n\n[来源取舍] 内部向量知识库命中置信度较高，网络搜索结果仅供参考，以向量库结果为准。"

    collected = "\n\n---\n\n".join(collected_parts)

    system_prompt = main_agent_content.get("system_prompt", "")
    synthesize_prompt = f"""请基于以下各来源的信息，回答用户的原始问题。

用户原始问题: {task_query}{authority_note}

{collected}

请用清晰的结构回答，引用具体数据，给出明确的结论。"""

    full: list[str] = []
    last_usage = None
    async for chunk in model.astream([
        SystemMessage(content=system_prompt),
        HumanMessage(content=synthesize_prompt),
    ]):
        if chunk.content:
            full.append(str(chunk.content))
            monitor.stream_chunk(str(chunk.content))
        if chunk.usage_metadata is not None:
            last_usage = chunk.usage_metadata
    monitor.stream_done()
    add_usage(last_usage)

    if full:
        return "".join(full)
    else:
        return ""


# =============================================================================
# 本地调试入口
# =============================================================================

if __name__ == "__main__":
    asyncio.run(
        run_deep_agent(
            "查一下阿莫西林的库存情况，然后从网上搜索阿莫西林的最新政策，汇总信息",
            "test_session_002",
        )
    )
