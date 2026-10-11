"""
端到端评测脚本（v1：只看结果，不看过程）

锚点从 gold DAG 换成「成功条件」：对每条样本调用 run_deep_agent() 端到端执行，
拿到最终 answer（文件类另读生成的文档），再用 LLM Judge 判 answer 是否满足
success_criteria。不规定路径——殊途同归自动被容纳。

Judge 输出 JSON {"pass": bool, "reason": str}，判定结果以 JSON 为准；
解析失败则降级为按关键词回退，仍失败记为 JUDGE_ERROR（不计入 PASS/FAIL）。

前提：子 agent 服务（8001/8002/8004）需在线。

用法:
    uv run python evaluate/eval_e2e.py --limit 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, SystemMessage

from shared.llm import model
from agents.orchestrator.main_agent import run_deep_agent

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 端到端执行依赖的子 agent 服务（与 agents/orchestrator/a2a_tools.py 的 SUBAGENT_URLS 对齐）
_REQUIRED_SERVICES = {
    "网络搜索 (8001)": "http://localhost:8001",
    "数据库查询 (8002)": "http://localhost:8002",
    "向量检索 (8004)": "http://localhost:8004",
}

_JUDGE_SYSTEM = (
    "你是一名严格的评测员。你的任务是判断「答案」是否满足给定的「成功条件」。"
    "只关注最终结果是否达到条件，不要评价获取答案的过程或路径。"
    '严格输出 JSON，格式：{"pass": true/false, "reason": "一句话说明"}，不要输出任何其他内容。'
)


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1 :].rstrip()
        if t.endswith("```"):
            t = t[:-3].rstrip()
    return t.strip()


def _parse_judge(raw: str) -> tuple[bool | None, str]:
    """解析 Judge 输出，返回 (pass, reason)。pass=None 表示解析失败。"""
    cleaned = _strip_fence(raw)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "pass" in obj:
            return bool(obj["pass"]), str(obj.get("reason", ""))
    except Exception:
        pass
    # 回退：找 "pass": true/false
    m = re.search(r'"pass"\s*:\s*(true|false)', cleaned, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true", cleaned[:200]
    return None, cleaned[:200]


async def _check_services() -> list[str]:
    """探测依赖的子 agent 服务是否可达，返回离线服务列表。

    服务离线时评测会静默产出「无法获取数据」式答案，被 Judge 判 false，
    造成大量误报。这里在开跑前做一次连通性检查，直接指出缺失的服务。
    """
    offline: list[str] = []
    for name, url in _REQUIRED_SERVICES.items():
        host = url.split("://", 1)[1].split(":", 1)[0]
        port = int(url.rsplit(":", 1)[1])
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=3
            )
            writer.close()
            await writer.wait_closed()
        except Exception:
            offline.append(f"{name} ({url})")
    return offline


async def _judge(criteria: str, answer: str) -> tuple[bool | None, str]:
    if not answer.strip():
        return False, "答案为空"
    prompt = f"## 成功条件\n{criteria}\n\n## 待评测答案\n{answer}"
    resp = await model.ainvoke([
        SystemMessage(content=_JUDGE_SYSTEM),
        HumanMessage(content=prompt),
    ])
    raw = str(resp.content) if resp.content else ""
    return _parse_judge(raw)


def _collect_files(session_id: str) -> str:
    """收集该 session 生成的文档内容（md 全文，pdf 只记文件名）。"""
    d = PROJECT_ROOT / "output" / f"session_{session_id}"
    if not d.exists():
        return ""
    parts: list[str] = []
    for f in sorted(d.glob("*.md")):
        try:
            parts.append(f"[文档 {f.name}]\n" + f.read_text(encoding="utf-8")[:4000])
        except Exception:
            pass
    for f in sorted(d.glob("*.pdf")):
        parts.append(f"[PDF文件] {f.name}")
    return "\n\n".join(parts)


async def eval_one(row: dict) -> dict:
    query = row["query"]
    criteria = row["success_criteria"]
    kind = row.get("kind", "qa")
    session_id = f"eval_{row['id']}_{int(time.time())}"

    t0 = time.time()
    answer = ""
    error = None
    try:
        answer = await run_deep_agent(query, session_id)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    files = _collect_files(session_id) if kind == "file" else ""
    judge_input = answer if not files else f"{answer}\n\n{files}"

    passed = None
    reason = ""
    if error:
        passed = False
        reason = f"执行失败: {error}"
    else:
        passed, reason = await _judge(criteria, judge_input)

    elapsed = time.time() - t0
    return {
        "id": row["id"],
        "kind": kind,
        "query": query,
        "passed": passed,
        "reason": reason,
        "elapsed": elapsed,
        "answer_preview": (answer or "")[:120].replace("\n", " "),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="端到端评测（只看结果）")
    parser.add_argument("--data", default=str(Path(__file__).parent / "data" / "tasks_success.jsonl"))
    parser.add_argument("--limit", type=int, default=0, help="只评测前 N 条（0 表示全部）")
    parser.add_argument("--ids", default="", help="只评测指定 id，逗号分隔（如 dag_010,dag_011）")
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        rows = [r for r in rows if r["id"] in wanted]
    elif args.limit:
        rows = rows[: args.limit]

    # 跳过缺少 success_criteria 的样本（如误用 tasks_dag.jsonl，那是 eval_plan 的数据）
    missing = [r.get("id", "?") for r in rows if not r.get("success_criteria")]
    if missing:
        print(f"[提示] {len(missing)} 条样本缺少 success_criteria 字段，已跳过: {', '.join(missing[:5])}"
              + ("..." if len(missing) > 5 else ""))
        print("[提示] tasks_dag.jsonl 是规划评测数据（配合 eval_plan.py）；端到端评测请用 tasks_success.jsonl")
        rows = [r for r in rows if r.get("success_criteria")]
    if not rows:
        print("无可评测样本，退出")
        return

    # 开跑前检查子 agent 服务在线情况，避免服务离线时静默产出大量 false 误报
    offline = await _check_services()
    if offline:
        print("\n[错误] 以下子 agent 服务不可达，端到端评测无法进行：")
        for s in offline:
            print(f"  - {s}")
        print("请先通过 `uv run python start_services.py` 启动全部服务后再评测。")
        return

    results = []
    for i, row in enumerate(rows, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(rows)}] 端到端执行: {row['query'][:30]}...")
        results.append(await eval_one(row))

    n = len(results)
    pass_n = sum(1 for r in results if r["passed"] is True)
    fail_n = sum(1 for r in results if r["passed"] is False)
    err_n = sum(1 for r in results if r["passed"] is None)

    print("\n" + "=" * 74)
    print("  端到端评测报告（只看结果）")
    print("=" * 74)
    print(f"  样本数    : {n}")
    print(f"  PASS      : {pass_n}")
    print(f"  FAIL      : {fail_n}")
    print(f"  JUDGE错误 : {err_n}")
    if pass_n + fail_n:
        print(f"  PASS 率   : {pass_n}/{pass_n + fail_n} = {pass_n / (pass_n + fail_n):.1%}")
    print("-" * 74)
    for r in results:
        mark = "PASS" if r["passed"] is True else ("FAIL" if r["passed"] is False else "JUDGE_ERROR")
        print(f"  [{mark}] {r['id']}  {r['kind']:<4} {r['query'][:24]}... ({r['elapsed']:.0f}s)")
        if r["reason"]:
            print(f"         ↳ {r['reason'][:160]}")
    print("=" * 74)


if __name__ == "__main__":
    asyncio.run(main())
