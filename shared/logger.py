import logging
import os
import sys


def setup_logging(level: int | str | None = None) -> None:
    """统一配置 root logger，所有模块通过 logging.getLogger(__name__) 继承。

    特性：
    - 输出到 stderr（兼容 MCP Server 的 stdout 协议约束）
    - Windows 下强制 stderr UTF-8 编码（避免 GBK 无法编码 emoji/中文）
    - 级别优先级：LOG_LEVEL 环境变量 > 参数 > 默认 INFO
    """
    if level is None:
        env_level = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    # Windows GBK 编码无法输出 Unicode/emoji，必须在 Windows 上 reconfigure
    if sys.platform == "win32":
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(level)
    # 避免重复添加 handler（start_services.py 多进程场景）
    if not root.handlers:
        root.addHandler(handler)
