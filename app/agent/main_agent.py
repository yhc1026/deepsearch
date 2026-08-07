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

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.llm import model

logger = logging.getLogger(__name__)
from app.agent.planner import Planner, Plan
from app.agent.executor import DAGExecutor
from app.agent.prompts import main_agent_content
from app.api.context import (
    reset_session_context,
    set_session_context,
    set_thread_context,
)
from app.api.monitor import monitor

project_root_path = Path(__file__).parents[1].resolve()

# 工具名简写，打印更短
_TOOL_ABBREV = {
    "call_network_search":  "网络搜索",
    "call_database_query":  "数据库",
    "call_ragflow_query":   "RAGFlow",
    "read_file_content":    "读附件",
    "generate_markdown":    "生成MD",
    "convert_md_to_pdf":    "转PDF",
}


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


async def run_deep_agent(task_query: str, session_id: str):
    """三阶段流水线执行主智能体任务。

    Phase 1: 规划 → Planner 将用户请求转换为结构化 DAG 计划
    Phase 2: 执行 → Executor 按拓扑顺序分批调度子 Agent
    Phase 3: 汇总 → LLM 整合结果生成最终交付物
    """
    logger.info(f"开始执行会话，session_id={session_id}")

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
    monitor.report_session_dir(session_dir_str)

    # ---- 当前日期上下文（供 Planner 将相对时间转为绝对日期） ----
    today = datetime.now()
    date_context = (
        f"\n[当前日期] {today.strftime('%Y年%m月%d日')} (星期{['一','二','三','四','五','六','日'][today.weekday()]})"
        f"\n[工作目录] {relative_session_dir_str}"
        f"{uploaded_context}"
    )

    full_query = task_query + date_context

    try:
        # ================================================================
        # Phase 1: 规划
        # ================================================================
        planner = Planner()
        plan: Plan = await planner.plan(full_query)

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

        # ================================================================
        # Phase 2: 执行
        # ================================================================
        executor = DAGExecutor()
        results: dict[str, str] = await executor.execute(plan)

        # ================================================================
        # Phase 3: 汇总
        # ================================================================
        final_answer = await _synthesize(task_query, plan, results)

        if final_answer:
            monitor.report_task_result(final_answer)

    except asyncio.CancelledError:
        monitor.report_task_cancelled()
        raise
    except Exception as e:
        monitor._emit("error", f"主智能体执行异常：{str(e)}")
        raise
    finally:
        reset_session_context(session_dir_token, session_id_token)


# =============================================================================
# Phase 3: 汇总器
# =============================================================================


async def _synthesize(task_query: str, plan: Plan, results: dict[str, str]) -> str:
    """汇总所有步骤结果，生成最终交付物。

    如果计划中已包含 generate_markdown / convert_md_to_pdf 步骤，
    这些步骤在 Phase 2 执行器中已完成（含 LLM 内容合成），
    这里只需返回简洁的执行摘要。

    如果计划中没有文件生成步骤（纯问答场景），
    则调用 LLM 整合所有数据收集结果，生成文本答案。
    """
    # 检查是否有文件生成步骤已完成
    file_steps_done = [
        s.id for s in plan.steps
        if s.tool in ("generate_markdown", "convert_md_to_pdf")
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
        if s.tool in ("call_network_search", "call_database_query", "call_ragflow_query")
    ]

    if not info_steps:
        # 没有信息获取步骤，直接返回第一个结果
        return results.get(plan.steps[0].id, "任务已完成") if plan.steps else "任务已完成"

    # 构造汇总 prompt
    collected = "\n\n---\n\n".join(
        f"### {s.description} (来源: {s.id})\n{results.get(s.id, '(无结果)')}"
        for s in info_steps
    )

    system_prompt = main_agent_content.get("system_prompt", "")
    synthesize_prompt = f"""请基于以下各来源的信息，回答用户的原始问题。

用户原始问题: {task_query}

{collected}

请用清晰的结构回答，引用具体数据，给出明确的结论。"""

    response = await model.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=synthesize_prompt),
    ])

    return str(response.content) if response.content else ""


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
