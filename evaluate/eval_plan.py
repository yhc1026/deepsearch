"""
DAG Plan 评测脚本（v1：确定性四层，无需对齐、无需 LLM judge）

对每条评测样本调用 Planner.plan() 生成 pred plan，与 gold plan 对比。
每条样本给出 PASS/FAIL 判定 + 中文错因；汇总输出正确率与各层指标。

判定规则（一条样本 PASS 需同时满足）：
1. 计划生成成功（JSON 解析 + 静态校验通过）
2. 每个 gold step 的工具方向被覆盖（tool 或 tool_alt 出现在 pred 工具集）
3. 没有多余工具（pred 里没有 gold 未认可的工具）
4. step 数落在 gold [min_steps, max_steps] 带宽内
5. kind=fallback 的样本，pred 需含 fallback_for 非空的兜底步骤

用法:
    uv run python evaluate/eval_plan.py --data evaluate/data/tasks_dag.jsonl
    uv run python evaluate/eval_plan.py --limit 3   # 先冒烟跑 3 条
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.orchestrator.planner import Planner

_ABBREV = {
    "call_network_search": "网络",
    "call_database_query": "数据库",
    "call_vector_search": "向量",
    "call_document_writer": "文档",
    "read_file_content": "读附件",
}


def _abbrev(tool: str) -> str:
    return _ABBREV.get(tool, tool)


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------


def tool_f1(gold_steps: list[dict], pred_steps: list) -> tuple[float, float, float]:
    """主工具的多重集对比，返回 (f1, precision, recall)。"""
    g = Counter(s["tool"] for s in gold_steps)
    p = Counter(s.tool for s in pred_steps)
    if not g and not p:
        return 1.0, 1.0, 1.0
    match = sum(min(g[t], p[t]) for t in set(g) | set(p))
    precision = match / sum(p.values()) if sum(p.values()) else 0.0
    recall = match / sum(g.values()) if sum(g.values()) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1, precision, recall


def _acceptable_set(gold_steps: list[dict]) -> set[str]:
    acc: set[str] = set()
    for s in gold_steps:
        acc.add(s["tool"])
        acc.update(s.get("tool_alt") or [])
    return acc


def _missing_dirs(gold_steps: list[dict], pred_steps: list) -> list[str]:
    """gold 中未被 pred 覆盖的工具方向（软命中失败的那些 step）。"""
    pred_tools = Counter(s.tool for s in pred_steps)
    miss = []
    for s in gold_steps:
        acc = {s["tool"]} | set(s.get("tool_alt") or [])
        if not any(t in pred_tools for t in acc):
            miss.append(_abbrev(s["tool"]))
    return miss


def _extra_tools(gold_steps: list[dict], pred_steps: list) -> list[str]:
    """pred 中出现、但 gold 任何 step 都不认可的工具。"""
    acc = _acceptable_set(gold_steps)
    return [t for t in dict.fromkeys(s.tool for s in pred_steps) if t not in acc]


# ---------------------------------------------------------------------------
# 单条评测
# ---------------------------------------------------------------------------


async def eval_one(planner: Planner, row: dict) -> dict:
    query = row["query"]
    gold_steps = row["steps"]
    kind = row["kind"]
    band = f"{row['min_steps']}~{row['max_steps']}"

    pred_steps = []
    valid = False
    error = None
    pred_goal = ""
    try:
        plan = await planner.plan(query)
        valid = True
        pred_steps = plan.steps
        pred_goal = plan.goal
    except Exception as exc:  # JSON 解析失败 / PlanValidationError / 网络等
        error = f"{type(exc).__name__}: {exc}"

    n_pred = len(pred_steps)
    granularity_ok = valid and row["min_steps"] <= n_pred <= row["max_steps"]
    f1, prec, rec = tool_f1(gold_steps, pred_steps)

    fallback_ok = None
    if kind == "fallback":
        fallback_ok = valid and any(s.fallback_for for s in pred_steps)

    missing = _missing_dirs(gold_steps, pred_steps) if valid else []
    extra = _extra_tools(gold_steps, pred_steps) if valid else []

    # ---- 错因汇总 → verdict ----
    reasons: list[str] = []
    if not valid:
        reasons.append(f"计划生成失败: {error}")
    else:
        if missing:
            reasons.append("工具缺失: " + "、".join(missing))
        if extra:
            reasons.append("多余工具: " + "、".join(_abbrev(t) for t in extra))
        if not granularity_ok:
            reasons.append(f"粒度超范围: gold {band}，pred {n_pred} 步")
        if fallback_ok is False:
            reasons.append("缺少兜底步骤")
    verdict = "PASS" if not reasons else "FAIL"

    return {
        "id": row["id"],
        "kind": kind,
        "query": query,
        "verdict": verdict,
        "reasons": reasons,
        "valid": valid,
        "granularity_ok": granularity_ok,
        "fallback_ok": fallback_ok,
        "gold_tools": [_abbrev(s["tool"]) + ("*" if s.get("fallback_for") else "") for s in gold_steps],
        "pred_tools": [_abbrev(s.tool) + ("*" if s.fallback_for else "") for s in pred_steps],
        "band": band,
        "n_pred": n_pred,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="DAG Plan 评测")
    parser.add_argument("--data", default=str(Path(__file__).parent / "data" / "tasks_dag.jsonl"))
    parser.add_argument("--limit", type=int, default=0, help="只评测前 N 条（0 表示全部）")
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    planner = Planner()
    results = []
    for i, row in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] 规划中: {row['query'][:26]}...", flush=True)
        results.append(await eval_one(planner, row))

    n = len(results)
    pass_n = sum(1 for r in results if r["verdict"] == "PASS")
    valid_n = sum(1 for r in results if r["valid"])
    gran_n = sum(1 for r in results if r["granularity_ok"])
    fallback_rows = [r for r in results if r["fallback_ok"] is not None]
    fallback_n = sum(1 for r in fallback_rows if r["fallback_ok"])

    avg_f1 = sum(r["f1"] for r in results) / n

    print()
    print("=" * 74)
    print("  DAG Plan 评测报告（v1 确定性判定）")
    print("=" * 74)
    print(f"  样本数       : {n}")
    print(f"  PASS 率      : {pass_n}/{n} = {pass_n/n:.1%}")
    print(f"  valid rate   : {valid_n}/{n} = {valid_n/n:.1%}")
    print(f"  粒度命中率   : {gran_n}/{n} = {gran_n/n:.1%}")
    print(f"  工具 F1(macro): {avg_f1:.3f}")
    print(f"  fallback 命中 : {fallback_n}/{len(fallback_rows)}" if fallback_rows else "  fallback 命中 : -")
    print("-" * 74)

    for r in results:
        g = "+".join(r["gold_tools"]) or "(空)"
        p = "+".join(r["pred_tools"]) or "(空)"
        print(f"  [{r['verdict']}] {r['id']}  {r['kind']:<15} {g} -> {p}")
        if r["reasons"]:
            for c in r["reasons"]:
                print(f"         ↳ {c}")
    print("=" * 74)

    # 追加一行 DAG Plan 评测记录到 eval_report.txt（以 [PLAN] 标记区分召回评测）
    report_path = Path(__file__).parent / "eval_report.txt"
    new_file = not report_path.exists()
    with open(report_path, "a", encoding="utf-8") as f:
        if new_file:
            f.write(
                "时间 | 类型 | 数据文件 | 样本 | PASS率 | valid率 | 粒度命中率 | "
                "工具F1 | 兜底命中\n"
            )
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_name = Path(args.data).name
        fb = f"{fallback_n}/{len(fallback_rows)}" if fallback_rows else "-"
        line = (
            f"{ts} | [PLAN] | {data_name} | {n} | {pass_n / n:.1%} | "
            f"{valid_n / n:.1%} | {gran_n / n:.1%} | {avg_f1:.3f} | {fb}"
        )
        f.write(line + "\n")


if __name__ == "__main__":
    asyncio.run(main())
