"""
文档摄入模块

支持纯文本、Markdown、PDF 的解析与分块，
将文档拆分为语义连贯的 chunk 后写入 ChromaDB。

依赖: pypdf (项目已有), langchain.text_splitter (项目已有)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .settings import CHUNK_OVERLAP, CHUNK_SIZE
from .vector_store import vector_store


def _read_file(file_path: str) -> str:
    """读取文件内容，根据扩展名选择解析器。"""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return _read_pdf(file_path)
    elif ext in (".txt", ".md", ".markdown", ".yml", ".yaml", ".json", ".csv"):
        return Path(file_path).read_text(encoding="utf-8", errors="replace")
    elif ext == ".docx":
        return _read_docx(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def _read_pdf(file_path: str) -> str:
    """读取 PDF 文件文本内容。"""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("请安装 pypdf: uv add pypdf")

    reader = PdfReader(file_path)
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _read_docx(file_path: str) -> str:
    """读取 Word 文档文本内容。"""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("请安装 python-docx: uv add python-docx")

    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def _split_text(text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """将长文本按语义边界切分为 chunk 列表。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", " ", ""],
        keep_separator=True,
    )
    docs = splitter.create_documents([text])
    chunks: list[dict[str, Any]] = []
    base_meta = metadata or {}
    for i, doc in enumerate(docs):
        chunks.append({
            "text": doc.page_content,
            "metadata": {**base_meta, "chunk_index": i},
        })
    return chunks


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


def ingest_file(
    file_path: str,
    collection_name: str = "default",
    metadata: dict[str, Any] | None = None,
) -> int:
    """将单个文件解析、分块并写入知识库。

    返回写入的 chunk 数量。
    """
    filename = os.path.basename(file_path)
    full_text = _read_file(file_path)
    base_meta = {
        "source": filename,
        "file_path": file_path,
        **(metadata or {}),
    }
    chunks = _split_text(full_text, base_meta)
    return vector_store.add_chunks(collection_name, chunks)


def ingest_text(
    text: str,
    collection_name: str = "default",
    metadata: dict[str, Any] | None = None,
) -> int:
    """将纯文本内容分块并写入知识库。

    返回写入的 chunk 数量。
    """
    chunks = _split_text(text, metadata)
    return vector_store.add_chunks(collection_name, chunks)


def ingest_directory(
    dir_path: str,
    collection_name: str = "default",
    extensions: tuple[str, ...] = (".pdf", ".txt", ".md", ".docx"),
) -> dict[str, int]:
    """批量导入目录下的所有支持文件。

    返回 {filename: chunk_count} 的统计字典。
    """
    stats: dict[str, int] = {}
    dir_path_obj = Path(dir_path)
    if not dir_path_obj.is_dir():
        raise NotADirectoryError(f"目录不存在: {dir_path}")

    for file_path_obj in dir_path_obj.rglob("*"):
        if not file_path_obj.is_file():
            continue
        if file_path_obj.suffix.lower() not in extensions:
            continue
        try:
            count = ingest_file(str(file_path_obj), collection_name)
            stats[file_path_obj.name] = count
        except Exception:
            stats[file_path_obj.name] = -1  # 标记失败
    return stats
