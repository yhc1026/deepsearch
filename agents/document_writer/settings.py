"""文档撰写配置加载。"""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

# 请求元数据未携带 session_dir 时的兜底输出目录
DEFAULT_OUTPUT_DIR = Path(os.getenv("DOCUMENT_OUTPUT_DIR", "./output")).resolve()
