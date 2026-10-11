"""
子 Agent / 工具统一业务结果信封。

约定（JSON 字符串或 dict）：
{
  "code": "HIT" | "MISS" | "ERROR",
  "content": "可供下游汇总的正文"
}

- HIT: 命中有效信息
- MISS: 正常执行但无可用知识（应触发 fallback）
- ERROR: 调用/服务异常（应触发 fallback）
"""

from __future__ import annotations

import json
from typing import Any, Optional

# 业务状态码
HIT = "HIT"
MISS = "MISS"
ERROR = "ERROR"

FALLBACK_CODES = frozenset({MISS, ERROR})

_VALID = frozenset({HIT, MISS, ERROR})


def make_result(code: str, content: str = "", score: Optional[float] = None) -> str:
    """构造标准结果信封（JSON 字符串）。

    可选 score 字段携带检索置信度（如向量 rerank 分数），供编排器在多来源
    冲突时做权重取舍；None 表示无置信度信号。
    """
    if code:
        code = code.upper()
    else:
        code = ERROR
    if code not in _VALID:
        code = ERROR
    if content:
        content_value = content
    else:
        content_value = ""
    payload = {"code": code, "content": content_value}
    if score is not None:
        payload["score"] = score
    return json.dumps(payload, ensure_ascii=False)


def parse_result(raw: Any) -> tuple[str, str]:
    """解析信封，返回 (code, content)。

    无法解析时：空 → ERROR；有文本 → 默认 HIT（兼容尚未改造的子 agent）。
    """
    if raw is None:
        return ERROR, ""

    if isinstance(raw, dict):
        code = str(raw.get("code", "")).upper()
        raw_content = raw.get("content", raw.get("result", ""))
        if raw_content:
            content = str(raw_content)
        else:
            content = ""
        if code in _VALID:
            return code, content
        if content:
            return HIT, content
        else:
            if raw:
                fallback_content = str(raw)
            else:
                fallback_content = ""
            return ERROR, fallback_content

    text = str(raw).strip()
    if not text:
        return ERROR, ""

    # 整体即 JSON
    data = _try_load_json(text)
    if data is not None:
        code = str(data.get("code", "")).upper()
        raw_content = data.get("content", data.get("result", ""))
        if raw_content:
            content = str(raw_content)
        else:
            content = ""
        if code in _VALID:
            return code, content

    # 正文中夹带 JSON
    json_text = _find_json_object(text)
    if json_text:
        data = _try_load_json(json_text)
        if data is not None:
            data_code = str(data.get("code", "")).upper()
            if data_code in _VALID:
                code = str(data["code"]).upper()
                raw_content = data.get("content", data.get("result", ""))
                if raw_content:
                    content = str(raw_content)
                else:
                    content = ""
                if content:
                    return code, content
                else:
                    return code, text

    # 兼容旧协议：无信封时当 HIT（有正文）
    return HIT, text


def parse_score(raw: Any) -> Optional[float]:
    """从信封中提取 score 字段；缺失或非法时返回 None。"""
    data: Optional[dict] = None
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str):
        data = _try_load_json(raw)
    if not isinstance(data, dict):
        return None
    score = data.get("score")
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def extract_envelope_code(raw: Any) -> Optional[str]:
    """仅当 raw 是/包含显式信封（带合法 code 字段的 JSON）时返回该 code，否则 None。

    与 parse_result 的区别：parse_result 对无信封文本默认返回 HIT；
    本函数对无信封文本返回 None，用于区分「子 agent 显式判定的 code」与「默认 HIT」。
    """
    if isinstance(raw, dict):
        code = str(raw.get("code", "")).upper()
        return code if code in _VALID else None

    text = str(raw).strip()
    if not text:
        return None

    data = _try_load_json(text)
    if data is not None:
        code = str(data.get("code", "")).upper()
        if code in _VALID:
            return code

    json_text = _find_json_object(text)
    if json_text:
        data = _try_load_json(json_text)
        if data is not None:
            code = str(data.get("code", "")).upper()
            if code in _VALID:
                return code

    return None


def needs_fallback(code: str) -> bool:
    """该业务码是否应触发兜底步骤。"""
    if code:
        normalized_code = code.upper()
    else:
        normalized_code = ERROR
    return normalized_code in FALLBACK_CODES


def _find_json_object(text: str) -> Optional[str]:
    """从文本中查找第一个完整的 JSON 对象字符串。"""
    start = text.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        start = text.find("{", start + 1)
    return None


def _try_load_json(text: str) -> Optional[dict]:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(data, dict):
        return data
    else:
        return None
