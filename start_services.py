"""
DeepSearch Agents 一键启动脚本 (单机多进程版)

启动 4 个独立 Agent 服务:
  - port 8000: 主智能体 (Orchestrator + Query Rewriter)
  - port 8001: 网络搜索智能体 (Tavily)
  - port 8002: 数据库查询智能体 (MySQL)
  - port 8003: RAGFlow 智能体 (RAGFlow SDK)

使用方式:
  uv run python start_services.py              # 生产模式，无热重载
  uv run python start_services.py --reload     # 开发模式，代码变更自动重启

前置条件: 已通过 `uv sync` 安装项目依赖。

关闭方式: Ctrl+C (所有子进程自动终止)
"""

import shutil
import subprocess
import sys
import time
from typing import List, Tuple

# 启动前检查 uv 是否可用
if shutil.which("uv") is None:
    sys.exit("错误: 未找到 uv，请先安装 uv 并运行 `uv sync` 安装依赖。")

# 服务列表: (名称, uvicorn app 路径, 端口)
SERVICES: List[Tuple[str, str, int]] = [
    ("主智能体",          "app.api.server:app",                       8000),
    ("网络搜索智能体",    "app.services.network_search_service:app",  8001),
    ("数据库查询智能体",  "app.services.database_query_service:app",  8002),
    ("RAGFlow智能体",     "app.services.ragflow_service:app",         8003),
]

processes: List[Tuple[str, subprocess.Popen]] = []


def start_service(name: str, app_path: str, port: int, reload: bool = False) -> subprocess.Popen:
    """通过 uv run 启动一个 uvicorn 子进程，确保使用项目虚拟环境中的依赖"""
    cmd = [
        "uv", "run", "uvicorn", app_path,
        "--host", "0.0.0.0",
        "--port", str(port),
    ]
    if reload:
        cmd.append("--reload")
    # 必须继承主进程 stdout：若用 PIPE 又不读，缓冲写满后子进程会阻塞假死
    proc = subprocess.Popen(
        cmd,
        stdout=None,
        stderr=None,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    processes.append((name, proc))
    reload_tag = " [reload]" if reload else ""
    print(f"  [{name}]{reload_tag} 启动中... http://localhost:{port}  (pid={proc.pid})")
    return proc


def shutdown() -> None:
    """优雅关闭所有子进程"""
    print("\n正在关闭所有服务...")
    for name, proc in reversed(processes):
        if proc.poll() is None:
            print(f"  停止 [{name}] pid={proc.pid}")
            proc.terminate()
    # 等待子进程结束
    for name, proc in processes:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"  强制终止 [{name}]")
            proc.kill()
    print("所有服务已关闭。")


if __name__ == "__main__":
    reload_mode = "--reload" in sys.argv

    print("=" * 55)
    print("  DeepSearch Agents 单机多进程版 启动中...")
    if reload_mode:
        print("  [开发模式] 启用热重载 (--reload)")
    print("=" * 55)
    print()

    # 启动所有服务
    for name, app_path, port in SERVICES:
        start_service(name, app_path, port, reload=reload_mode)

    print()
    print("所有服务已启动！")
    print()
    print("  端点总览:")
    print(f"    {'主智能体 (WebSocket+HTTP)':<30} http://localhost:8000")
    print(f"    {'网络搜索智能体 A2A':<30} http://localhost:8001")
    print(f"    {'数据库查询智能体 A2A':<30} http://localhost:8002")
    print(f"    {'RAGFlow智能体 A2A':<30} http://localhost:8003")
    print()
    print("  前端连接: http://localhost:8000")
    print("  Agent Cards: GET http://localhost:800{1,2,3}/")
    if reload_mode:
        print()
        print("  [reload] 源码变更时各服务自动重启")
    print()
    print("  按 Ctrl+C 停止所有服务")
    print()

    try:
        # 主进程等待，同时监控子进程状态
        while True:
            time.sleep(2)
            # 检查是否有子进程意外退出
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"  ! [{name}] 意外退出 (exitcode={proc.returncode})")
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()
