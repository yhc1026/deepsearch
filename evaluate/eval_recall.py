"""
向量数据库召回评测脚本

复刻生产检索管线（search_knowledge_base）但保留 chunk id，按评测数据的 kind 分类报告：
  - 有答案条目（exact_single / multi_gold / cross_doc / vague）：算召回
      · 粗排候选池命中率、Recall@1、Recall@3、MRR
  - 无答案条目（no_answer）：算误召回率（检索不应返回内容却返回了）

两个关键处理：
1. gold 与检索结果都按 chunk「内容指纹」去重——库中同一文档被重复摄入会产生内容
   相同、id 不同的重复 chunk，直接比 id 会误判召回。
2. 直接复用 retrieval_tools 的内部函数（query 变体 / 去重 / 分数融合），
   保证评测逻辑与生产完全一致，且不侵入生产代码。

用法:
    uv run python evaluate/eval_recall.py --data evaluate/data/sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# 使脚本可从任意目录运行：将项目根加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.vector_search.rerank import rerank
from agents.vector_search.retrieval_tools import (
    _deduplicate_results,
    _fuse_scores,
    _generate_query_variants,
)
from agents.vector_search.settings import (
    HYBRID_TOP_K,
    QUERY_VARIANTS,
    RERANK_CANDIDATE_K,
    RERANK_THRESHOLD,
    RERANK_TOP_K,
)
from agents.vector_search.vector_store import vector_store


def retrieve(query: str, collection: str) -> tuple[list[str], list[str], bool]:
    """复刻 search_knowledge_base 检索流程，返回 (粗排id, 精排id, rerank是否成功)。"""
    variants = _generate_query_variants(query, n=QUERY_VARIANTS)

    all_results = []
    for v in variants:
        all_results.extend(
            vector_store.hybrid_search(collection, v, top_k=HYBRID_TOP_K)
        )

    merged = _deduplicate_results(all_results)
    candidates = merged[:RERANK_CANDIDATE_K]
    coarse_ids = [c["id"] for c in candidates]

    if not candidates:
        return coarse_ids, [], False

    candidate_texts = [c["text"] for c in candidates]
    rerank_ok = True
    try:
        rerank_scores = {
            int(r["index"]): float(r["relevance_score"])
            for r in rerank(query, candidate_texts)
        }
    except Exception:
        rerank_scores = {}
        rerank_ok = False

    final = _fuse_scores(candidates, rerank_scores)
    if rerank_ok:
        final = [c for c in final if c.get("rerank_score", 0.0) >= RERANK_THRESHOLD]
    fine_ids = [c["id"] for c in final[:RERANK_TOP_K]]
    return coarse_ids, fine_ids, rerank_ok


def _fp(text: str) -> str:
    """chunk 内容指纹：前 200 字符（与生产去重逻辑 text[:200] 一致）。"""
    return text[:200]


def _content_hits(
    gold_ids: list[str], retrieved_ids: list[str], id_to_text: dict, k: int | None = None
) -> float:
    """内容级召回：gold 答案点（去重）中被检索结果（去重）覆盖的比例。

    k=None 表示用全部 retrieved_ids，否则取前 k 个。
    """
    gold_fps = {_fp(id_to_text[g]) for g in gold_ids if g in id_to_text}
    if not gold_fps:
        return 0.0
    ret = retrieved_ids if k is None else retrieved_ids[:k]
    ret_fps = {_fp(id_to_text[r]) for r in ret if r in id_to_text}
    return len(gold_fps & ret_fps) / len(gold_fps)


def _content_mrr(gold_ids: list[str], retrieved_ids: list[str], id_to_text: dict) -> float:
    """内容级 MRR：gold 答案点第一次出现的排名的倒数（按内容去重后的顺序）。"""
    gold_fps = {_fp(id_to_text[g]) for g in gold_ids if g in id_to_text}
    if not gold_fps:
        return 0.0
    seen: set[str] = set()
    rank = 0
    for r in retrieved_ids:
        fp = _fp(id_to_text.get(r, ""))
        if fp in seen:
            continue
        seen.add(fp)
        rank += 1
        if fp in gold_fps:
            return 1.0 / rank
    return 0.0


def _content_rank(gold_ids: list[str], retrieved_ids: list[str], id_to_text: dict) -> int | None:
    """返回 gold 答案点在去重后的检索结果中第一次出现的排名（1 起），未命中返回 None。"""
    gold_fps = {_fp(id_to_text[g]) for g in gold_ids if g in id_to_text}
    seen: set[str] = set()
    rank = 0
    for r in retrieved_ids:
        fp = _fp(id_to_text.get(r, ""))
        if fp in seen:
            continue
        seen.add(fp)
        rank += 1
        if fp in gold_fps:
            return rank
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="向量数据库召回评测")
    parser.add_argument("--data", required=True, help="评测数据 JSONL 路径")
    args = parser.parse_args()

    rows: list[dict] = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("评测数据为空")
        return

    # 加载 id -> text 映射（按数据中出现的 collection）
    collections = {r.get("collection", "default") for r in rows}
    id_to_text: dict[str, str] = {}
    for col in collections:
        for ch in vector_store.get_chunks(col):
            id_to_text[ch["id"]] = ch["text"]

    stats: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "coarse": 0.0, "r1": 0.0, "r3": 0.0, "mrr": 0.0}
    )
    no_answer_n = 0
    no_answer_retrieved = 0
    rerank_all_ok = True
    details: list[tuple] = []

    pbar = tqdm(rows, desc="评测进度", ncols=100)
    for r in pbar:
        kind = r.get("kind", "exact_single")
        query = r["query"]
        collection = r.get("collection", "default")
        gold = r.get("gold_chunk_ids", [])

        pbar.set_postfix_str(query[:18])
        coarse_ids, fine_ids, rerank_ok = retrieve(query, collection)
        rerank_all_ok = rerank_all_ok and rerank_ok

        if kind == "no_answer":
            no_answer_n += 1
            if fine_ids:
                no_answer_retrieved += 1
            details.append((kind, query, None, None, bool(fine_ids)))
            continue

        coarse_hit = _content_hits(gold, coarse_ids, id_to_text)
        r1 = _content_hits(gold, fine_ids, id_to_text, k=1)
        r3 = _content_hits(gold, fine_ids, id_to_text, k=3)
        m = _content_mrr(gold, fine_ids, id_to_text)

        s = stats[kind]
        s["n"] += 1
        s["coarse"] += 1.0 if coarse_hit > 0 else 0.0
        s["r1"] += r1
        s["r3"] += r3
        s["mrr"] += m

        rank = _content_rank(gold, fine_ids, id_to_text)
        details.append((kind, query, coarse_hit > 0, rank, None))

    # ---- 汇总有答案条目 ----
    total_n = sum(s["n"] for s in stats.values())
    total_coarse = sum(s["coarse"] for s in stats.values())
    total_r1 = sum(s["r1"] for s in stats.values())
    total_r3 = sum(s["r3"] for s in stats.values())
    total_mrr = sum(s["mrr"] for s in stats.values())

    def pct(x: float, n: int) -> str:
        return f"{x / n:.1%}" if n else "-"

    print("=" * 62)
    print("  向量数据库召回评测报告")
    print("=" * 62)
    print(f"  数据条数    : {len(rows)} (有答案 {total_n} / 无答案 {no_answer_n})")
    print(f"  粗排候选池  : {RERANK_CANDIDATE_K}    精排 top-k : {RERANK_TOP_K}")
    print("  [说明] gold 与结果按 chunk 内容去重后比对（库中存在重复摄入的 chunk）")
    if not rerank_all_ok:
        print("  [警告] rerank 精排失败，已降级为纯粗排排序")
    print("-" * 62)
    print(f"  [总体-有答案 {total_n} 条]")
    print(f"    粗排候选池命中率 : {pct(total_coarse, total_n)}")
    print(f"    Recall@1         : {pct(total_r1, total_n)}")
    print(f"    Recall@3         : {pct(total_r3, total_n)}")
    print(f"    MRR              : {total_mrr / total_n:.3f}" if total_n else "    MRR              : -")
    print(f"  [总体-无答案 {no_answer_n} 条]")
    print(f"    误召回率         : {pct(no_answer_retrieved, no_answer_n)}")
    print("-" * 62)
    print("  分 kind:")
    print(f"  {'kind':<14}{'条数':>5}{'粗排命中':>10}{'Recall@1':>10}{'Recall@3':>10}{'MRR':>8}")
    for kind in ["exact_single", "multi_gold", "cross_doc", "vague"]:
        s = stats[kind]
        n = s["n"]
        if n == 0:
            continue
        print(
            f"  {kind:<14}{n:>5}{pct(s['coarse'], n):>10}{pct(s['r1'], n):>10}"
            f"{pct(s['r3'], n):>10}{s['mrr'] / n:>8.3f}"
        )
    if no_answer_n:
        print(
            f"  {'no_answer':<14}{no_answer_n:>5}{'-':>10}{'-':>10}"
            f"{'误召回 ' + pct(no_answer_retrieved, no_answer_n):>12}{'-':>6}"
        )
    print("-" * 62)
    print("  明细:")
    by_kind = defaultdict(list)
    for d in details:
        by_kind[d[0]].append(d)
    for kind in ["exact_single", "multi_gold", "cross_doc", "vague", "no_answer"]:
        items = by_kind.get(kind, [])
        if not items:
            continue
        print(f"  [{kind}]")
        for i, (_, query, coarse_hit, rank, _retrieved) in enumerate(items, 1):
            if kind == "no_answer":
                flag = "误召回" if _retrieved else "正确拒答"
                print(f"    {i:2d}. {flag}  {query}")
            else:
                coarse = "命中" if coarse_hit else "未中"
                fine = f"@{rank}" if rank else "未中"
                print(f"    {i:2d}. 粗排{coarse} 精排{fine}  {query}")
    print("=" * 62)

    # 追加一行评测记录到 eval_report.txt
    report_path = Path(__file__).parent / "eval_report.txt"
    new_file = not report_path.exists()
    with open(report_path, "a", encoding="utf-8") as f:
        if new_file:
            f.write(
                "时间 | 数据文件 | 总数 | 有答案 | 无答案 | 粗排命中率 | Recall@1 | "
                "Recall@3 | MRR | 误召回率\n"
            )
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_name = Path(args.data).name
        mrr_str = f"{total_mrr / total_n:.3f}" if total_n else "-"
        line = (
            f"{ts} | {data_name} | {len(rows)} | {total_n} | {no_answer_n} | "
            f"{pct(total_coarse, total_n)} | {pct(total_r1, total_n)} | "
            f"{pct(total_r3, total_n)} | {mrr_str} | {pct(no_answer_retrieved, no_answer_n)}"
        )
        f.write(line + "\n")


if __name__ == "__main__":
    main()
