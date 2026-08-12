"""
提示词配置加载模块 (共享)

负责读取各 agent 目录下的 prompts.yml 配置。
每个 agent 自包含其 prompt 定义，路径为 agents/{name}/prompts.yml。
"""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(file_path: Path) -> dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# 当前文件位于 shared/prompts.py，parents[1] 即项目根目录 deepsearch/
project_root = Path(__file__).parents[1]


def load_agent_prompts(agent_name: str) -> dict[str, Any]:
    """加载指定 agent 目录下的 prompts.yml"""
    path = project_root / "agents" / agent_name / "prompts.yml"
    return load_yaml(path)


# 向后兼容：保持原有变量名，各模块无需修改 import
main_agent_content = load_agent_prompts("orchestrator")
sub_agents_content = {
    "tavily": load_agent_prompts("network_search"),
    "db": load_agent_prompts("database_query"),
    # "ragflow": load_agent_prompts("ragflow_search"),  # TODO: 取消注释以启用 RAGFlow
    "vector_search": load_agent_prompts("vector_search"),
}
