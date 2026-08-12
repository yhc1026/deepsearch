"""
向量知识库资料装载工具

用法:
  uv run python agents/vector_search/ingest_data.py <文件或目录> [--collection 知识库名]

示例:
  uv run python agents/vector_search/ingest_data.py ./docs/                    # 导入整个目录
  uv run python agents/vector_search/ingest_data.py ./docs/report.pdf          # 导入单个文件
  uv run python agents/vector_search/ingest_data.py ./policies/ -c policies    # 导入到指定知识库
  uv run python agents/vector_search/ingest_data.py --list                     # 查看已有知识库
  uv run python agents/vector_search/ingest_data.py --stats                    # 查看各知识库统计
  uv run python agents/vector_search/ingest_data.py --peek -c default          # 浏览知识库内容
  uv run python agents/vector_search/ingest_data.py --peek -c default -n 20    # 浏览前20条
"""

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

# Windows 终端 GBK 编码兼容：强制 stdout 使用 UTF-8，无法编码的字符替换为 ?
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.vector_search.ingestion import ingest_directory, ingest_file
from agents.vector_search.vector_store import vector_store


def main():
    parser = argparse.ArgumentParser(description="向量知识库资料装载工具")
    parser.add_argument("path", nargs="?", help="要导入的文件或目录路径")
    parser.add_argument("-c", "--collection", default="default", help="目标知识库名称（默认 default）")
    parser.add_argument("--list", action="store_true", help="列出所有知识库")
    parser.add_argument("--stats", action="store_true", help="查看各知识库统计")
    parser.add_argument("--delete", metavar="COLLECTION", help="删除指定知识库")
    parser.add_argument("--peek", action="store_true", help="浏览知识库中的文档块内容")
    parser.add_argument("-n", "--limit", type=int, default=10, help="浏览时显示的条数（默认 10）")
    args = parser.parse_args()

    if args.list:
        collections = vector_store.list_collections()
        if not collections:
            print("当前没有任何知识库")
        else:
            print(f"共 {len(collections)} 个知识库:")
            for c in collections:
                stats = vector_store.collection_stats(c)
                print(f"  {c} — {stats.get('chunk_count', 0)} 个文本块")
        return

    if args.stats:
        collections = vector_store.list_collections()
        if not collections:
            print("当前没有任何知识库")
        else:
            for c in collections:
                stats = vector_store.collection_stats(c)
                print(f"  {c}: {stats.get('chunk_count', 0)} chunks")
        return

    if args.delete:
        vector_store.delete_collection(args.delete)
        print(f"知识库 [{args.delete}] 已删除")
        return

    if args.peek:
        _peek_collection(args.collection, args.limit)
        return

    if not args.path:
        parser.print_help()
        return

    source = Path(args.path)
    if not source.exists():
        print(f"错误: 路径不存在 — {args.path}")
        sys.exit(1)

    if source.is_file():
        print(f"导入文件: {source} → 知识库「{args.collection}」")
        count = ingest_file(str(source), collection_name=args.collection)
        print(f"完成: {count} 个文本块已写入")

    elif source.is_dir():
        print(f"导入目录: {source} -> 知识库 [{args.collection}]")
        stats = ingest_directory(str(source), collection_name=args.collection)
        total = 0
        failed = 0
        for fname, cnt in stats.items():
            if cnt < 0:
                print(f"  FAIL {fname}")
                failed += 1
            else:
                print(f"  OK   {fname} - {cnt} chunks")
                total += cnt
        print(f"Done: {total} chunks, {failed} files failed")

    print(f"\n知识库「{args.collection}」统计: {vector_store.collection_stats(args.collection)}")


def _peek_collection(collection_name: str, limit: int) -> None:
    """浏览知识库中已入库的文档块内容。"""
    try:
        collection = vector_store._client.get_collection(name=collection_name)
    except Exception:
        print(f"知识库 [{collection_name}] 不存在")
        return

    data = collection.get(limit=limit)
    if not data["documents"]:
        print(f"知识库 [{collection_name}] 为空")
        return

    print(f"知识库 [{collection_name}] 共 {collection.count()} 条, 显示前 {limit} 条:")
    print("=" * 70)

    for i, (doc_id, doc_text, meta) in enumerate(
        zip(data["ids"], data["documents"], data["metadatas"])
    ):
        source = meta.get("source", "-") if meta else "-"
        chunk_idx = meta.get("chunk_index", "-") if meta else "-"
        print(f"\n--- [{i + 1}] source={source}  chunk={chunk_idx} ---")
        print(f"    id: {doc_id}")
        print(f"    text: {doc_text[:500]}{'...' if len(doc_text) > 500 else ''}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
