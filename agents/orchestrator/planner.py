"""
规划器模块

负责在任务执行前，调用 LLM 将用户请求转换为结构化的 DAG 执行计划。
计划产出后经静态校验（依赖合法性、工具存在性、规则约束），交给执行器按序执行。

核心理念：LLM 当参谋（产出计划），代码当执行者（保证计划被严格执行）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from shared.llm import model
from shared.prompts import main_agent_content


# =============================================================================
# 数据结构
# =============================================================================


class PlanStep(BaseModel):
    """计划中的一个步骤"""

    id: str = Field(description="步骤唯一标识，如 step_1, step_2")
    tool: str = Field(description="工具函数名，如 call_network_search")
    description: str = Field(description="本步骤要做什么，中文简述")
    query: str = Field(description="自包含的完整查询语句/操作指令")
    depends_on: list[str] = Field(default_factory=list, description="依赖的前置步骤 id 列表")
    fallback_for: Optional[str] = Field(default=None, description="作为哪个步骤的兜底（仅在该步骤返回'匹配度过低'等失败结果时执行）")


class Plan(BaseModel):
    """完整的执行计划"""

    goal: str = Field(description="任务目标的一句话描述")
    steps: list[PlanStep] = Field(description="按依赖关系排列的执行步骤列表")


class PlanValidationError(Exception):
    """计划校验失败"""

    pass


@dataclass
class ToolInfo:
    """工具元信息，供规划器了解可用工具"""

    name: str
    description: str
    category: str = "info"  # "info" | "file" | "local"


# =============================================================================
# 规划器
# =============================================================================


class Planner:
    """DAG 计划生成器。

    工作流程：
    1. 构造 planning prompt（工具列表 + few-shot + 用户请求）
    2. 调用 LLM，要求输出 JSON 格式的 Plan
    3. 解析并校验计划合法性
    4. 返回 Plan 对象
    """

    # LLM 有时会产出用 ```json 包裹的 JSON，需要去掉围栏

    def __init__(self, tools: Optional[list[ToolInfo]] = None):
        if tools is not None:
            self.tools = tools
        else:
            self.tools = _default_tools()

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------

    async def plan(self, task_query: str) -> Plan:
        """给定用户请求，生成并校验执行计划。"""
        prompt = self._build_planning_prompt(task_query)
        raw_json = await self._invoke_planner_llm(prompt)
        plan = self._parse_plan(raw_json)
        self._validate(plan)
        return plan

    # ------------------------------------------------------------------
    # Prompt 构造
    # ------------------------------------------------------------------

    def _build_planning_prompt(self, task_query: str) -> str:
        tools_desc = self._format_tools()
        planning_config = main_agent_content.get("planning", {})
        system_prompt = planning_config.get("system_prompt", _DEFAULT_PLANNING_SYSTEM_PROMPT)
        examples = planning_config.get("examples", _DEFAULT_EXAMPLES)

        examples_text = ""
        for i, ex in enumerate(examples, 1):
            examples_text += f"\n### 示例 {i}\n**用户请求**: {ex['request']}\n**计划**:\n```json\n{json.dumps(ex['plan'], ensure_ascii=False, indent=2)}\n```\n"

        return f"""{system_prompt}

{examples_text}

## 可用工具

{tools_desc}

## 当前任务

用户请求: {task_query}

请输出 JSON 计划（不要包含任何非 JSON 内容）："""

    @staticmethod
    def _format_tools() -> str:
        lines: list[str] = []
        for t in _default_tools():
            lines.append(f"- **{t.name}** [{t.category}]: {t.description}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM 调用
    # ------------------------------------------------------------------

    async def _invoke_planner_llm(self, prompt: str) -> str:
        """调用 LLM 产出计划，带 basic retry 处理格式问题。"""
        messages = [
            SystemMessage(
                content="你是一个任务规划专家。你的唯一职责是输出严格合法的 JSON，不得输出任何其他内容。"
            ),
            HumanMessage(content=prompt),
        ]

        # 使用较短的超时和低温度保证输出稳定
        response = await model.ainvoke(messages)
        if response.content:
            content: str = str(response.content)
        else:
            content = ""

        # 去掉可能的 markdown 围栏
        content = _strip_json_fence(content)

        return content

    # ------------------------------------------------------------------
    # 解析
    # ------------------------------------------------------------------

    def _parse_plan(self, raw_json: str) -> Plan:
        """将 LLM 输出的原始 JSON 字符串解析为 Plan 对象。"""
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise PlanValidationError(f"计划 JSON 格式无效: {exc}") from exc

        try:
            return Plan.model_validate(data)
        except ValidationError as exc:
            raise PlanValidationError(f"计划结构不符合规范: {exc}") from exc

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------

    def _validate(self, plan: Plan) -> None:
        """静态校验计划的合法性，不通过则抛 PlanValidationError。"""
        step_ids = {s.id for s in plan.steps}
        available_tools = {t.name for t in self.tools}
        info_tools = {t.name for t in self.tools if t.category == "info"}
        file_tools = {t.name for t in self.tools if t.category == "file"}

        # 空计划是合法的——表示 LLM 判断不需要调用任何工具，直接口头回复
        if not plan.steps:
            return

        for step in plan.steps:
            # 1. 工具存在性
            if step.tool not in available_tools:
                raise PlanValidationError(
                    f"步骤 {step.id} 引用了不存在的工具 '{step.tool}'，可用: {available_tools}"
                )

            # 2. 依赖合法性
            for dep in step.depends_on:
                if dep not in step_ids:
                    raise PlanValidationError(
                        f"步骤 {step.id} 依赖了不存在的步骤 '{dep}'"
                    )
                if dep == step.id:
                    raise PlanValidationError(f"步骤 {step.id} 不能依赖自身")

            # 3. 循环依赖检测
            self._check_cycle(plan, step)

        # 4. 规则校验：文件生成工具必须在最后，且依赖所有信息获取步骤
        info_step_ids = [s.id for s in plan.steps if s.tool in info_tools]
        file_steps = [s for s in plan.steps if s.tool in file_tools]
        for fs in file_steps:
            for info_id in info_step_ids:
                if info_id not in fs.depends_on:
                    raise PlanValidationError(
                        f"步骤 {fs.id}（文件生成）必须依赖所有信息获取步骤，"
                        f"缺少对 {info_id} 的依赖"
                    )

    def _check_cycle(self, plan: Plan, step: PlanStep, visited: Optional[set] = None) -> None:
        """DFS 检测循环依赖。"""
        if visited is not None:
            current_visited = visited
        else:
            current_visited = set()
        if step.id in current_visited:
            raise PlanValidationError(f"检测到循环依赖，涉及步骤: {step.id}")
        current_visited.add(step.id)
        for dep_id in step.depends_on:
            dep_step = next(s for s in plan.steps if s.id == dep_id)
            self._check_cycle(plan, dep_step, current_visited.copy())


def _strip_json_fence(content: str) -> str:
    """去掉 LLM 输出中常见的 markdown JSON 围栏。"""
    stripped = content.strip()
    if not stripped.startswith("```"):
        return content

    first_newline = stripped.find("\n")
    if first_newline == -1:
        return content

    inner = stripped[first_newline + 1 :]
    inner_stripped = inner.rstrip()
    if inner_stripped.endswith("```"):
        inner_stripped = inner_stripped[:-3].rstrip()
    return inner_stripped


# =============================================================================
# 默认工具列表 & 提示词
# =============================================================================


def _default_tools() -> list[ToolInfo]:
    """主智能体当前可用的工具清单。"""
    return [
        # 远程信息获取工具
        ToolInfo(
            name="call_ragflow_query",
            category="info",
            description=(
                "查询 RAGFlow 内部知识库，覆盖行业研报、市场分析、政策解读、"
                "产品资料、制度文件等专业领域知识。涉及行业趋势、市场研究、"
                "专业知识问答时优先使用。参数 query: 自包含的检索问题。"
            ),
        ),
        ToolInfo(
            name="call_database_query",
            category="info",
            description=(
                "查询企业结构化数据库（药品信息、库存、销售记录等业务数据）。"
                "参数 query: 自然语言描述的数据需求（不要写 SQL），"
                "包含数据维度、筛选条件、聚合方式。"
            ),
        ),
        ToolInfo(
            name="call_network_search",
            category="info",
            description=(
                "从互联网检索公开信息（实时新闻、天气、百科、最新事件）。"
                "仅用于内部知识库和数据库无法覆盖的补充查询。"
                "参数 query: 自包含的完整搜索语句。"
            ),
        ),
        # 本地文件工具
        ToolInfo(
            name="read_file_content",
            category="local",
            description=(
                "读取用户上传的附件内容。参数 filename: 文件名，"
                "instruction: 读取指示（如'提取关键信息'）。"
            ),
        ),
        ToolInfo(
            name="generate_markdown",
            category="file",
            description=(
                "生成 Markdown 文档。参数 content: 文档内容（markdown 格式），"
                "filename: 文件名（含 .md 后缀），path: 保存路径。"
                "该工具必须放在所有信息获取步骤之后。"
            ),
        ),
        ToolInfo(
            name="convert_md_to_pdf",
            category="file",
            description=(
                "将 Markdown 文件转换为 PDF。依赖 generate_markdown 先生成 md 文件。"
                "参数 md_filename: 源 md 文件名，pdf_filename: 目标 pdf 文件名。"
            ),
        ),
    ]


_DEFAULT_PLANNING_SYSTEM_PROMPT = """你是一个资深的任务规划专家。你的职责是将用户的复杂请求分解为一个结构化的 DAG 执行计划。

## 输出格式

严格输出以下 JSON 结构，不得包含任何其他内容：

```json
{
  "goal": "一句话描述任务目标",
  "steps": [
    {
      "id": "step_1",
      "tool": "工具函数名",
      "description": "本步骤要做什么",
      "query": "自包含的完整查询语句（必须具体化时间、消解指代、补全维度）",
      "depends_on": []
    }
  ]
}
```

## 规划规则

1. **Query 自包含**：每个 step 的 query 必须是一个完整的独立语句，子 agent 仅凭 query 就能理解和执行。必须：
   - 将相对时间转为绝对日期（从上下文推断当前日期）
   - 消解所有指代（"我们公司/竞品/那个" → 具体实体）
   - 补全查询维度（如查询库存 → 药品名+批号+数量+仓库+效期）

2. **依赖声明**：
   - 无依赖的步骤 `depends_on: []`，会并行执行
   - B 需要 A 的结果时，B 的 `depends_on` 必须包含 A 的 id
   - 在 B 的 query 中用 `{{step_X}}` 引用 A 的完整结果，或用 `{{step_X.field_name}}` 引用 A 结果中的特定字段

3. **顺序约束**：
   - 信息获取工具（call_ragflow_query / call_database_query / call_network_search）必须在文件生成之前
   - 文件生成工具（generate_markdown）必须放在所有信息获取步骤之后，且 depends_on 必须包含所有信息获取步骤
   - convert_md_to_pdf 必须依赖 generate_markdown 的步骤

4. **最大化并行**：能在同一批并行执行的步骤就不要串行。

5. **步骤数量**：简单任务 1-3 步，复杂任务 3-8 步。如果用户只是打招呼、闲聊、询问能力范围，不需要调用任何工具，返回空的 steps 列表即可。

6. **知识库优先 + 强制兜底**：
   - 涉及行业分析、市场研究、政策解读时优先用 call_ragflow_query
   - 任何使用 call_ragflow_query 的步骤，**必须**附带一个 fallback_for 指向它的 call_network_search 兜底步骤
   - 任何使用 call_database_query 查询「百科类/知识类」信息（如药品功效、代谢周期、适用症状等非结构化知识）的步骤，**必须**附带一个 fallback_for 指向它的 call_network_search 兜底步骤
   - 兜底步骤的 depends_on 必须包含被兜底的步骤 id，query 中引用 {{被兜底步骤id}} 来知道搜索目标（但要忽略匹配度过低的文本，提取其中的实体名称自行搜索）
   - 兜底步骤的 tool 统一用 call_network_search

7. **允许在一个计划中多次调用同一个工具**，只要这些调用的目的不同。
"""

_DEFAULT_EXAMPLES = [
    {
        "request": "你好 / hi / 今天心情怎么样 / 你能做什么",
        "plan": {
            "goal": "对话交流",
            "steps": [],
        },
    },
    {
        "request": "帮我查一下阿莫西林的库存情况，并生成一份库存报告。",
        "plan": {
            "goal": "查询阿莫西林库存并生成报告",
            "steps": [
                {
                    "id": "step_1",
                    "tool": "call_database_query",
                    "description": "查询阿莫西林库存数据",
                    "query": "查询药品名称包含'阿莫西林'的所有批次的库存数量、批号、生产日期、效期、仓库位置、供应商信息",
                    "depends_on": [],
                },
                {
                    "id": "step_2",
                    "tool": "generate_markdown",
                    "description": "根据库存数据生成报告",
                    "query": "根据以下阿莫西林库存数据生成一份库存分析报告，包含：库存总览、批次明细、效期预警、仓库分布。\n\n库存数据：{{step_1}}",
                    "depends_on": ["step_1"],
                },
            ],
        },
    },
    {
        "request": "分析阿莫西林的市场供需情况，结合我们的库存数据给出采购建议，生成 PDF 报告。",
        "plan": {
            "goal": "综合市场分析和库存数据，生成阿莫西林采购建议 PDF 报告",
            "steps": [
                {
                    "id": "step_1",
                    "tool": "call_ragflow_query",
                    "description": "检索阿莫西林行业分析和市场供需趋势",
                    "query": "阿莫西林原料药市场规模、供需格局、价格走势、主要生产厂家分析报告",
                    "depends_on": [],
                },
                {
                    "id": "step_2",
                    "tool": "call_database_query",
                    "description": "查询阿莫西林当前库存和近半年销售数据",
                    "query": "查询药品名称包含'阿莫西林'的所有批次库存数量、批号、效期；以及近6个月（2026年2月至2026年8月）的销售数量和金额，按月份汇总",
                    "depends_on": [],
                },
                {
                    "id": "step_3",
                    "tool": "generate_markdown",
                    "description": "汇总市场分析和库存数据，生成采购建议报告",
                    "query": "基于以下数据生成一份阿莫西林采购建议报告（不少于1000字），包含：1.市场供需分析 2.当前库存状况 3.销售趋势 4.采购建议（含建议采购量和时间节点） 5.风险提示。\n\n市场分析：{{step_1}}\n\n库存与销售数据：{{step_2}}",
                    "depends_on": ["step_1", "step_2"],
                },
                {
                    "id": "step_4",
                    "tool": "convert_md_to_pdf",
                    "description": "将 Markdown 报告转为 PDF",
                    "query": "将 step_3 生成的报告转换为 PDF",
                    "depends_on": ["step_3"],
                },
            ],
        },
    },
    {
        "request": "查一下今天北京天气怎么样，温度多少？",
        "plan": {
            "goal": "查询北京今日天气",
            "steps": [
                {
                    "id": "step_1",
                    "tool": "call_network_search",
                    "description": "查询北京今日天气",
                    "query": "北京 2026年8月6日 天气 气温 降水 湿度 风速 空气质量",
                    "depends_on": [],
                }
            ],
        },
    },
    {
        "request": "找出库存最多的药品，然后查一下它是不是OTC药品。",
        "plan": {
            "goal": "找出库存最多的药品并判断其OTC属性",
            "steps": [
                {
                    "id": "step_1",
                    "tool": "call_database_query",
                    "description": "查询库存数量最多的药品",
                    "query": "查询所有药品的库存数量，按库存量降序排列，找出库存最多的药品名称、批号和库存量，只返回TOP 1",
                    "depends_on": [],
                },
                {
                    "id": "step_2",
                    "tool": "call_ragflow_query",
                    "description": "查询step_1找到的药品是否为OTC",
                    "query": "{{step_1}} 中提到的药品是否属于OTC（非处方药）？请说明其分类和相关信息",
                    "depends_on": ["step_1"],
                },
                {
                    "id": "step_2_fallback",
                    "tool": "call_network_search",
                    "description": "RAGFlow失败时兜底搜索该药品的OTC属性",
                    "query": "{{step_1}} 中提到的主要药品名称 是否为OTC非处方药 药品分类",
                    "depends_on": ["step_2"],
                    "fallback_for": "step_2",
                },
            ],
        },
    },
    {
        "request": "查询一下库存最多的药品，代谢周期是多久",
        "plan": {
            "goal": "查询库存最多药品的代谢周期",
            "steps": [
                {
                    "id": "step_1",
                    "tool": "call_database_query",
                    "description": "查询库存数量最多的药品",
                    "query": "查询所有药品的库存数量，按库存量降序排列，找出库存最多的药品名称，只返回TOP 1",
                    "depends_on": [],
                },
                {
                    "id": "step_2",
                    "tool": "call_ragflow_query",
                    "description": "在知识库中检索该药品的代谢周期",
                    "query": "{{step_1}} 中提到的主要药品的代谢周期是多长时间？半衰期是多少？",
                    "depends_on": ["step_1"],
                },
                {
                    "id": "step_2_fallback",
                    "tool": "call_network_search",
                    "description": "RAGFlow无结果时兜底搜索代谢周期",
                    "query": "{{step_1}} 中提到的主要药品名称 代谢周期 半衰期 多长时间",
                    "depends_on": ["step_2"],
                    "fallback_for": "step_2",
                },
            ],
        },
    },
]
