"""
DAG 执行器模块

按拓扑顺序分批执行 Plan 中的步骤，保证：
- 每个步骤都会被执行，不会遗漏
- 依赖关系严格执行，不会乱序
- 同批次内无依赖步骤并行执行
- 模板变量 {{step_id}} / {{step_id.field}} 自动填充

LLM 不参与执行决策——这是纯确定性代码。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

WHITE = "\033[37m"
RESET = "\033[0m"

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

from shared.llm import model
from agents.orchestrator.planner import Plan, PlanStep
from agents.orchestrator.context import get_session_context
from shared.monitor import monitor
from shared.agent_result import ERROR, HIT, needs_fallback, parse_result

# =============================================================================
# 工具注册表
# =============================================================================

from agents.orchestrator.a2a_tools import (
    call_database_query,
    call_network_search,
    # call_ragflow_query,  # TODO: 取消注释以启用 RAGFlow
    call_vector_search,
)
from agents.orchestrator.markdown_tools import generate_markdown
from agents.orchestrator.pdf_tools import convert_md_to_pdf
from agents.orchestrator.upload_file_read_tool import read_file_content

# 工具名 → LangChain BaseTool 实例
_TOOL_REGISTRY: dict[str, Any] = {
    "call_network_search": call_network_search,
    "call_database_query": call_database_query,
    # "call_ragflow_query": call_ragflow_query,  # TODO: 取消注释以启用 RAGFlow
    "call_vector_search": call_vector_search,
    "read_file_content": read_file_content,
    "generate_markdown": generate_markdown,
    "convert_md_to_pdf": convert_md_to_pdf,
}

# A2A 信息获取工具（query → {"query": query}）
_A2A_TOOLS = {"call_network_search", "call_database_query", "call_vector_search"}  # call_ragflow_query 已禁用


def _extract_template_step_ids(template: str) -> list[str]:
    """从模板字符串中提取 {{step_id}} 形式的步骤 id。"""
    ref_ids: list[str] = []
    index = 0
    while index < len(template):
        if template[index : index + 2] == "{{":
            cursor = index + 2
            step_chars: list[str] = []
            while cursor < len(template) and (
                template[cursor].isalnum() or template[cursor] == "_"
            ):
                step_chars.append(template[cursor])
                cursor += 1
            if cursor + 1 < len(template) and template[cursor : cursor + 2] == "}}":
                ref_ids.append("".join(step_chars))
                index = cursor + 2
                continue
        index += 1
    return ref_ids


def _replace_template_variables(query: str, results: dict[str, str]) -> str:
    """填充模板变量 {{step_id}} 和 {{step_id.field}}。"""
    if "{{" not in query:
        return query

    output_parts: list[str] = []
    index = 0
    while index < len(query):
        if query[index : index + 2] == "{{":
            cursor = index + 2
            step_chars: list[str] = []
            while cursor < len(query) and (
                query[cursor].isalnum() or query[cursor] == "_"
            ):
                step_chars.append(query[cursor])
                cursor += 1

            if cursor < len(query) and query[cursor] == ".":
                cursor += 1
                field_chars: list[str] = []
                while cursor < len(query) and (
                    query[cursor].isalnum() or query[cursor] == "_"
                ):
                    field_chars.append(query[cursor])
                    cursor += 1

            if cursor + 1 < len(query) and query[cursor : cursor + 2] == "}}":
                step_id = "".join(step_chars)
                if step_id not in results:
                    logger.warning(f"模板引用了尚未完成的步骤 {step_id}")
                    output_parts.append(query[index : cursor + 2])
                else:
                    output_parts.append(results[step_id])
                index = cursor + 2
                continue

        output_parts.append(query[index])
        index += 1

    return "".join(output_parts)


# =============================================================================
# 执行器
# =============================================================================


class ExecutionError(Exception):
    """执行失败"""
    pass


class DAGExecutor:
    """DAG 计划执行器。

    使用方式:
        executor = DAGExecutor()
        results = await executor.execute(plan)

    results 是一个 {step_id: result_string} 字典。
    """

    def __init__(self, tool_registry: dict[str, Any] | None = None):
        if tool_registry is not None:
            self._registry = tool_registry
        else:
            self._registry = _TOOL_REGISTRY

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    async def execute(
        self,
        plan: Plan,
        resume_state: dict[str, Any] | None = None,
        on_batch_done: Callable[[dict[str, str], dict[str, str], int], Awaitable[None]] | None = None,
    ) -> dict[str, str]:
        """执行整个计划，返回 {step_id: result_string}。

        resume_state: 断点恢复的初始状态 {"results", "result_codes", "batch_index"}，
                      从 batch_index+1 开始继续执行。
        on_batch_done: 每个 batch 完成后回调 (results, result_codes, batch_index)，
                      用于将执行进度落 checkpoint（异步，会被 await）。
        """
        if not plan.steps:
            return {}

        batches = self._topological_batches(plan)

        results: dict[str, str] = {}
        result_codes: dict[str, str] = {}
        start_batch = 0
        if resume_state:
            results = dict(resume_state.get("results") or {})
            result_codes = dict(resume_state.get("result_codes") or {})
            start_batch = int(resume_state.get("batch_index", -1)) + 1

        logger.info(
            f"计划共 {len(plan.steps)} 步，分 {len(batches)} 批执行"
            + (f"，从第 {start_batch + 1} 批恢复" if start_batch else "")
        )
        monitor._emit(
            "plan_start",
            f"开始执行计划，共 {len(plan.steps)} 步"
            + (f"（从第 {start_batch + 1} 批恢复）" if start_batch else ""),
            {
                "goal": plan.goal,
                "total_steps": len(plan.steps),
                "batches": len(batches),
                "resume_from": start_batch,
            },
        )

        for batch_idx in range(start_batch, len(batches)):
            batch = batches[batch_idx]
            batch_ids = [s.id for s in batch]
            logger.info(f"批次 {batch_idx + 1}/{len(batches)}: {batch_ids}")

            active_steps: list[PlanStep] = []
            for step in batch:
                if step.fallback_for:
                    target_code = result_codes.get(step.fallback_for, ERROR)
                    if not needs_fallback(target_code):
                        logger.info(
                            f"跳过兜底步骤 {step.id}（{step.fallback_for} code={target_code}）"
                        )
                        print(
                            f"{WHITE}  ⊘ 跳过 {step.tool} — {step.description}"
                            f"（{step.fallback_for} code={target_code}）{RESET}"
                        )
                        results[step.id] = (
                            f"[已跳过] 上游步骤 {step.fallback_for} 已命中(HIT)，无需兜底"
                        )
                        result_codes[step.id] = HIT
                        continue
                    print(
                        f"{WHITE}  → 调用 {step.tool} — {step.description}"
                        f" [兜底: {step.fallback_for} code={target_code}]{RESET}"
                    )
                else:
                    print(f"{WHITE}  → 调用 {step.tool} — {step.description}{RESET}")
                active_steps.append(step)

            if not active_steps:
                continue

            tasks = [self._execute_step(step, results) for step in active_steps]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for step, raw in zip(active_steps, batch_results):
                if isinstance(raw, Exception):
                    logger.error(f"步骤 {step.id} 失败: {raw}")
                    monitor._emit(
                        "step_error",
                        f"步骤 {step.id} 执行失败",
                        {"step_id": step.id, "error": str(raw)},
                    )
                    results[step.id] = f"[错误] {step.id} 执行失败: {raw}"
                    result_codes[step.id] = ERROR
                else:
                    code, content = parse_result(raw)
                    results[step.id] = content
                    result_codes[step.id] = code
                    logger.info(
                        f"步骤 {step.id} 完成 code={code} ({len(content)} 字符)"
                    )

            if on_batch_done:
                await on_batch_done(results, result_codes, batch_idx)

        monitor._emit(
            "plan_complete",
            f"计划执行完成，共 {len(plan.steps)} 步",
            {"completed_steps": len(results), "result_codes": result_codes},
        )
        return results

    # ------------------------------------------------------------------
    # 拓扑排序
    # ------------------------------------------------------------------

    @staticmethod
    def _topological_batches(plan: Plan) -> list[list[PlanStep]]:
        """Kahn's algorithm 变种：按层级分组，同层内步骤可并行执行。"""
        step_map = {s.id: s for s in plan.steps}
        in_degree: dict[str, int] = {s.id: len(s.depends_on) for s in plan.steps}
        dependents: dict[str, list[str]] = {s.id: [] for s in plan.steps}

        for s in plan.steps:
            for dep_id in s.depends_on:
                dependents[dep_id].append(s.id)

        batches: list[list[PlanStep]] = []
        current = [s for s in plan.steps if in_degree[s.id] == 0]

        while current:
            batches.append(current)
            next_batch: list[PlanStep] = []

            for s in current:
                for dependent_id in dependents[s.id]:
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        next_batch.append(step_map[dependent_id])

            current = next_batch

        processed = sum(len(b) for b in batches)
        if processed != len(plan.steps):
            unprocessed = [
                s.id for s in plan.steps
                if s.id not in {x.id for b in batches for x in b}
            ]
            raise ExecutionError(f"存在未处理的步骤（循环依赖?）: {unprocessed}")

        return batches

    # ------------------------------------------------------------------
    # 单步执行
    # ------------------------------------------------------------------

    async def _execute_step(self, step: PlanStep, results: dict[str, str]) -> str:
        """执行单个步骤：填充模板 → LLM 重写 query → 构造工具参数 → 调用工具。"""
        raw_query = self._fill_template(step.query, results)

        tool = self._registry.get(step.tool)
        if tool is None:
            raise ExecutionError(f"未知工具: {step.tool}")

        # 如果模板填充改变了 query（说明引用了上游结果），用 LLM 重写为精确查询
        has_template_refs = "{{" in step.query
        if has_template_refs and step.tool in _A2A_TOOLS:
            query = await self._rewrite_query(step.description, step.query, raw_query, results)
        else:
            query = raw_query

        # generate_markdown 需要额外一步 LLM 合成：把"指令+数据"转为正式内容
        if step.tool == "generate_markdown":
            query = await self._synthesize_content(query)

        params = self._build_tool_params(step.tool, query)

        monitor.report_tool(
            tool_name=step.tool,
            args={"params": params, "step_id": step.id,
                  "description": step.description},
        )

        # LangChain BaseTool 统一使用 ainvoke，内部自动处理 sync/async
        result = await tool.ainvoke(params)

        if result:
            return str(result)
        else:
            return ""

    # ------------------------------------------------------------------
    # 工具参数映射
    # ------------------------------------------------------------------

    def _build_tool_params(self, tool_name: str, query: str) -> dict:
        """将填充后的 query 字符串转换为工具所需的参数字典。

        不同工具有不同的参数签名，但 Planner 产出的 query 是统一字符串格式。
        这里做适配转换。
        """
        if tool_name in _A2A_TOOLS:
            return {"query": query}

        if tool_name == "read_file_content":
            return {"filename": query.strip(), "instruction": "提取关键内容"}

        if tool_name == "generate_markdown":
            session_dir = get_session_context()
            # 从 query 前 50 个字符推断文件名
            safe_title = self._safe_filename(query)
            filename = f"{safe_title}.md"
            return {
                "content": query,
                "filename": filename,
                "path": session_dir,
            }

        if tool_name == "convert_md_to_pdf":
            # query 应该是源 md 文件名
            md_name = query.strip().replace("\n", " ")
            if not md_name.endswith(".md"):
                md_name += ".md"
            pdf_name = md_name.replace(".md", ".pdf")
            return {
                "md_filename": md_name,
                "pdf_filename": pdf_name,
            }

        # fallback
        return {"query": query}

    @staticmethod
    def _safe_filename(text: str, max_len: int = 40) -> str:
        """从文本中提取安全的文件名片段。"""
        # 取第一行，去掉特殊字符
        first_line = text.strip().split("\n")[0]
        safe = "".join(c for c in first_line if c.isalnum()
                       or c in " _-").strip().replace(" ", "_")
        if len(safe) > max_len:
            safe = safe[:max_len]
        if safe:
            return safe
        else:
            return "output"

    # ------------------------------------------------------------------
    # LLM 内容合成（generate_markdown 专用）
    # ------------------------------------------------------------------

    async def _synthesize_content(self, instruction: str) -> str:
        """将「指令+数据」组合转为正式的文档内容。

        Plan 中的 generate_markdown 步骤，其 query 包含两部分：
        1. 文档生成的指令（如"生成一份XXX报告，包含章节A/B/C"）
        2. 通过 {{step_X}} 模板注入的上游数据

        填充后的文本是"指令+数据"的混合体，这里调用 LLM 将其转为
        结构化的正式 Markdown 文档内容。
        """
        system_prompt = (
            "你是一个专业的研究报告撰写助手。"
            "请根据用户的指令和提供的数据，生成一份结构完整的 Markdown 文档。"
            "要求：不少于 1000 字，章节结构清晰，数据引用准确，有明确结论。"
            "直接输出 Markdown 正文，不要输出任何解释性前言或后记。"
        )

        response = await model.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=instruction),
        ])

        if response.content:
            return str(response.content)
        else:
            return instruction

    # ------------------------------------------------------------------
    # LLM Query 重写（动态依赖解析）
    # ------------------------------------------------------------------

    async def _rewrite_query(
        self,
        description: str,
        template: str,
        raw_query: str,
        results: dict[str, str],
    ) -> str:
        """将模板填充后的粗糙 query 重写为精确的自包含查询。"""
        # 收集被引用的上游步骤结果
        ref_ids = _extract_template_step_ids(template)
        upstream_context = ""
        for rid in ref_ids:
            if rid in results:
                upstream_context += f"\n### 上游步骤 {rid} 的返回结果:\n{results[rid]}\n"

        prompt = f"""你是一个查询重写专家。你的任务是根据上游步骤的返回结果，将一个含有占位符的粗糙查询重写为精确的自包含查询。

## 本步骤的目的
{description}

## 原始查询模板（含 {{step_X}} 占位符）
{template}

## 上游步骤的完整返回结果
{upstream_context}

## 粗填充后的查询（你需要改写它）
{raw_query}

## 重写规则
1. 从上游结果中提取精确的关键值（如药品名、数量、日期等），替换模板占位符
2. 不要保留上游结果的冗长叙述，只提取事实数据
3. 改写后的 query 必须是自包含的完整查询，子 agent 仅凭 query 就能理解和执行
4. 如果上游结果没有返回具体值（比如返回了错误），保留原始模板结构但注明查询失败
5. 直接输出改写后的完整 query，不要输出任何解释"""

        response = await model.ainvoke([
            SystemMessage(content="你是查询重写专家。只输出改写后的 query，不要输出任何其他内容。"),
            HumanMessage(content=prompt),
        ])

        if response.content:
            rewritten = str(response.content).strip()
        else:
            rewritten = raw_query
        if rewritten != raw_query:
            logger.debug(f"Query 重写: {template[:60]}... → {rewritten[:80]}...")
        return rewritten

    # ------------------------------------------------------------------
    # 模板变量填充
    # ------------------------------------------------------------------

    def _fill_template(self, query: str, results: dict[str, str]) -> str:
        """填充模板变量 {{step_id}} 和 {{step_id.field}}。"""
        return _replace_template_variables(query, results)
