"""校验 evaluate/data/sample.jsonl 中每个 gold_chunk_id 都真实存在于向量库。

检查项：
1. 每个 gold id 是否在目标 collection 中（不存在即为致命错误）。
2. 多 gold 题的目标 chunk 是否确实含可回答问题的事实（打印文本片段供人工复核）。
3. docs 分布是否与真实语料比例大致一致。
4. 无答案题的 gold 是否为空。

用法: uv run python evaluate/validate_sample.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from chromadb.config import Settings

from agents.vector_search.settings import CHROMA_PERSIST_DIR

SAMPLE = Path(__file__).resolve().parent / "data" / "sample.jsonl"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    persist = CHROMA_PERSIST_DIR
    if not Path(persist).is_absolute():
        persist = str(project_root / persist)
    client = chromadb.PersistentClient(
        path=persist, settings=Settings(anonymized_telemetry=False)
    )

    store: dict[str, tuple[str, str]] = {}
    for col in client.list_collections():
        data = col.get()
        for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            store[cid] = (doc, str((meta or {}).get("source", "")))

    rows = [json.loads(l) for l in SAMPLE.read_text(encoding="utf-8").splitlines() if l.strip()]

    missing: list[tuple[str, str]] = []
    total_gold = 0
    gold_hit = 0
    doc_counter: Counter[str] = Counter()
    kind_counter: Counter[str] = Counter()

    for r in rows:
        kind_counter[r["kind"]] += 1
        doc_counter[r["doc"]] += 1
        for gid in r["gold_chunk_ids"]:
            total_gold += 1
            if gid in store:
                gold_hit += 1
            else:
                missing.append((r["query"], gid))

    print("=" * 60)
    print(f"  样本条数        : {len(rows)}")
    print(f"  题型分布        : {dict(kind_counter)}")
    print(f"  文档分布        : {dict(doc_counter)}")
    print(f"  gold id 总数    : {total_gold}")
    print(f"  真实存在        : {gold_hit}/{total_gold}  ({gold_hit/total_gold:.1%})")
    print(f"  多 gold 题数    : {sum(1 for r in rows if len(r['gold_chunk_ids'])>1)}")
    print(f"  无答案题数      : {sum(1 for r in rows if not r['gold_chunk_ids'])}")
    print("=" * 60)

    if missing:
        print(f"[致命] {len(missing)} 个 gold id 在库中不存在：")
        for q, gid in missing[:20]:
            print(f"  - {gid}  <-  {q}")
        sys.exit(1)

    # 逐条打印 gold 文本片段，供人工复核「该 chunk 确实能回答该 query」
    print("\n逐条复核（前 12 字片段）:")
    for i, r in enumerate(rows, 1):
        golds = r["gold_chunk_ids"]
        if not golds:
            print(f"\n{i:3d}. [无答案] {r['query']}")
            continue
        print(f"\n{i:3d}. [{r['kind']}] {r['query']}  (gold x{len(golds)})")
        seen_text: set[str] = set()
        for gid in golds:
            text = store[gid][0][:110].replace("\n", " ")
            if text in seen_text:
                continue
            seen_text.add(text)
            dup = " (含重复副本)" if golds.count(gid) > 1 else ""
            print(f"      · {text}{dup}")

    print("\n全部 gold id 校验通过。")


if __name__ == "__main__":
    main()
