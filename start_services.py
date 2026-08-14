"""
DeepSearch Agents 一键启动脚本 (单机多进程版)

启动 5 个独立 Agent 服务:
  - port 8000: 主智能体 (Orchestrator + Query Rewriter)
  - port 8001: 网络搜索智能体 (Tavily)
  - port 8002: 数据库查询智能体 (MySQL)
  - port 8003: RAGFlow 智能体 (RAGFlow SDK)
  - port 8004: 向量检索智能体 (ChromaDB + OpenAI Embedding)

使用方式:
  uv run python start_services.py              # 生产模式，无热重载
  uv run python start_services.py --reload     # 开发模式，代码变更自动重启

前置条件: 已通过 `uv sync` 安装项目依赖。

关闭方式: Ctrl+C (所有子进程自动终止)
"""

import logging
import shutil
import subprocess
import sys
import time
from typing import List, Tuple

# 在所有 import 之前屏蔽第三方库的弃用警告
import os
os.environ["PYTHONWARNINGS"] = "ignore"

from shared.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# 启动前检查 uv 是否可用
if shutil.which("uv") is None:
    sys.exit("错误: 未找到 uv，请先安装 uv 并运行 `uv sync` 安装依赖。")

# 服务列表: (名称, uvicorn app 路径, 端口)
SERVICES: List[Tuple[str, str, int | None]] = [
    ("主智能体",          "agents.orchestrator.server:app",                       8000),
    ("网络搜索智能体",    "agents.network_search.server:app",              8001),
    ("数据库查询智能体",  "agents.database_query.server:app",              8002),
    # ("RAGFlow智能体",     "agents.ragflow_search.server:app",                     8003),  # TODO: 取消注释以启用 RAGFlow
    ("向量检索智能体",    "agents.vector_search.server:app",               8004),
    ("长期记忆智能体",    "agents.memory_agent.server:app",              8005),
    ("MySQL MCP Server",  "agents.database_query.mcp_server:http_app", 8100),
    ("异步摘要智能体",     "agents.backend.summary_agent.server:main",    None),
]

processes: List[Tuple[str, subprocess.Popen]] = []
exited: set = set()  # 已退出并打印过的进程，避免每 2 秒重复刷屏


def start_service(name: str, app_path: str, port: int | None, reload: bool = False) -> subprocess.Popen:
    """启动一个子进程。port 为 None 时用 python -m，否则用 uvicorn。"""
    if port is None:
        # 非 HTTP 服务，如异步摘要 Agent，模块路径为 agents.backend.summary_agent.server
        module_path = app_path.rsplit(":", 1)[0]
        cmd = ["uv", "run", "python", "-m", module_path]
    else:
        cmd = [
            "uv", "run", "uvicorn", app_path,
            "--host", "0.0.0.0",
            "--port", str(port),
            "--no-access-log",
            "--log-level", "warning",
        ]
        if reload:
            cmd.append("--reload")
    # 必须继承主进程 stdout：若用 PIPE 又不读，缓冲写满后子进程会阻塞假死
    env = os.environ.copy()
    env["DEEPSEARCH_SERVICE_NAME"] = name
    proc = subprocess.Popen(
        cmd,
        stdout=None,
        stderr=None,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    processes.append((name, proc))
    return proc


def shutdown() -> None:
    """优雅关闭所有子进程"""
    logger.info("正在关闭所有服务...")
    for name, proc in reversed(processes):
        if proc.poll() is None:
            logger.info(f"停止 [{name}] pid={proc.pid}")
            proc.terminate()
    # 等待子进程结束
    for name, proc in processes:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(f"强制终止 [{name}]")
            proc.kill()
    logger.info("所有服务已关闭。")


if __name__ == "__main__":
    reload_mode = "--reload" in sys.argv

    for name, app_path, port in SERVICES:
        start_service(name, app_path, port, reload=reload_mode)

    try:
        # 主进程等待，同时监控子进程状态
        while True:
            time.sleep(2)
            # 检查是否有子进程意外退出
            for name, proc in processes:
                if proc.poll() is not None and name not in exited:
                    exited.add(name)
                    logger.warning(f"[{name}] 意外退出 (exitcode={proc.returncode}, pid={proc.pid})")
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()
