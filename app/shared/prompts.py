"""
提示词配置加载模块 (共享)

负责读取 app/prompt/prompts.yml 中的主智能体和子智能体配置。
主智能体和各子智能体服务都从这里导入配置。
"""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(file_path: Path) -> dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# 当前文件位于 app/shared/prompts.py，parents[1] 即 app 目录
app_root_path = Path(__file__).parents[1]
yaml_file_path = app_root_path / "prompt" / "prompts.yml"

prompt_yaml_content = load_yaml(yaml_file_path)

main_agent_content = prompt_yaml_content["main_agent"]
sub_agents_content = prompt_yaml_content["sub_agents"]
