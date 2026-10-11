"""从 chroma_db 导出真实 chunk 语料，供构造评测集使用。

输出 evaluate/data/corpus.jsonl，每行: {"id","source","chunk_index","text"}
按 source 分文件另存，便于逐文档阅读出题。
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

OUT_DIR = Path(__file__).resolve().parent / "data"
COLLECTION = "default"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    persist = CHROMA_PERSIST_DIR
    if not Path(persist).is_absolute():
        persist = str(project_root / persist)

    client = chromadb.PersistentClient(
        path=persist, settings=Settings(anonymized_telemetry=False)
    )
    col = client.get_collection(COLLECTION)
    data = col.get()

    rows: list[dict] = []
    for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        meta = meta or {}
        rows.append({
            "id": cid,
            "source": meta.get("source", "未知来源"),
            "chunk_index": meta.get("chunk_index", -1),
            "text": text,
        })

    rows.sort(key=lambda r: (r["source"], r["chunk_index"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = OUT_DIR / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 按 source 分文件，便于逐文档出题
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    for src, items in by_src.items():
        safe = src.replace(".pdf", "").replace("/", "_")
        with open(OUT_DIR / f"corpus_{safe}.jsonl", "w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"导出 {len(rows)} 个 chunk -> {corpus_path}")
    for src, items in sorted(by_src.items(), key=lambda x: -len(x[1])):
        avg = sum(len(i["text"]) for i in items) / len(items)
        print(f"  {len(items):4d}  {src}  (平均 {avg:.0f} 字符)")


if __name__ == "__main__":
    main()
