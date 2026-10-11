"""长期记忆召回参数配置。"""

import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

# dense 粗排召回候选数（进入 rerank 精排的池大小）
MEMORY_RECALL_CANDIDATE_K = int(os.getenv("MEMORY_RECALL_CANDIDATE_K", "20"))

# rerank 精排分数阈值：relevance_score 低于该值直接丢弃（BGE 输出约 0~1）
MEMORY_RECALL_THRESHOLD = float(os.getenv("MEMORY_RECALL_THRESHOLD", "0.3"))

# dense-only 降级阈值：rerank 不可用时对 dense 分数（1 - L2 距离）的近似阈值
MEMORY_RECALL_DENSE_THRESHOLD = float(os.getenv("MEMORY_RECALL_DENSE_THRESHOLD", "0.0"))

# 时间衰减半衰期（天）：final = score * exp(-age_days / half_life)
MEMORY_RECALL_HALF_LIFE_DAYS = float(os.getenv("MEMORY_RECALL_HALF_LIFE_DAYS", "90"))

# 来源多样性：同一 source_session_id 最多保留的条数
MEMORY_RECALL_MAX_PER_SESSION = int(os.getenv("MEMORY_RECALL_MAX_PER_SESSION", "2"))

# 最终注入 prompt 的记忆条数
MEMORY_RECALL_TOP_K = int(os.getenv("MEMORY_RECALL_TOP_K", "5"))
