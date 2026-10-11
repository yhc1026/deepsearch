"""
大模型初始化模块 (共享)

负责从 .env 中读取模型配置，并创建项目统一复用的模型对象。
主智能体和各子智能体服务都从这里导入 model。
"""

import os

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(find_dotenv(), override=True)

model = init_chat_model(
    model=os.getenv("MODEL"),
    model_provider="openai",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

# 自定义 base_url 时 stream_usage 默认不会开启，显式打开，
# 使 astream 的最后一个 chunk 携带 usage_metadata，供 token 统计读取。
model.stream_usage = True
