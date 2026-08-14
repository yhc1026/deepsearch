"""
异步摘要 Agent — 独立进程，定时扫描需要概括的会话，渐进式生成上下文摘要。

只概括第 1 ~ n-3 轮（保留最近 3 轮完整），写入：
- conversations.summary    → 单轮一句话摘要
- sessions.context_summary  → 全局渐进式上下文摘要
"""

import asyncio
import logging
import os
from pathlib import Path

import mysql.connector
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from shared.llm import model
from shared.logger import setup_logging
from shared.prompts import load_yaml

load_dotenv()
setup_logging()

logging.getLogger("__main__").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# 加载本 agent 的 prompt 模板
_prompts = load_yaml(Path(__file__).resolve().parent / "prompts.yml")

# 扫描间隔：启动时立即执行一次，之后每 24 小时扫描一次
SCAN_INTERVAL_SECONDS = 24 * 60 * 60


def _get_db_config() -> dict:
    config = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE"),
        "charset": os.getenv("MYSQL_CHARSET", "utf8mb4"),
        "collation": os.getenv("MYSQL_COLLATION", "utf8mb4_unicode_ci"),
        "autocommit": True,
        "connection_timeout": int(os.getenv("MYSQL_CONNECTION_TIMEOUT", "10")),
    }
    config = {k: v for k, v in config.items() if v is not None}
    for key in ("user", "password", "database"):
        if key not in config:
            raise ValueError(f"缺失数据库配置：{key}")
    return config


def _build_prompt(prompt: str, **kwargs) -> str:
    """安全填充 prompt 模板，截断过长字段。"""
    safe = {}
    for key, value in kwargs.items():
        if key == "assistant_result":
            safe[key] = (value or "")[:2000]
        elif key == "turns_text":
            safe[key] = (value or "")[:4000]
        elif key == "new_turns_text":
            safe[key] = (value or "")[:2000]
        else:
            safe[key] = value or ""
    return prompt.format(**safe)


async def _generate_turn_summary(user_query: str, assistant_result: str) -> str:
    """生成单轮对话的一句话摘要（≤60字）。"""
    try:
        prompt = _build_prompt(
            _prompts["turn_summary"],
            user_query=user_query,
            assistant_result=assistant_result,
        )
        response = await model.ainvoke([HumanMessage(content=prompt)])
        return str(response.content).strip()[:120]
    except Exception:
        logger.warning("生成轮次摘要失败", exc_info=True)
        return ""


async def _generate_context_summary(turns_text: str) -> str:
    """根据多轮对话文本，生成全局上下文摘要（≤200字）。"""
    try:
        prompt = _build_prompt(_prompts["context_summary"], turns_text=turns_text)
        response = await model.ainvoke([HumanMessage(content=prompt)])
        return str(response.content).strip()[:300]
    except Exception:
        logger.warning("生成上下文摘要失败", exc_info=True)
        return ""


async def _merge_context_summary(old_summary: str, new_turns_text: str) -> str:
    """将已有概要与新增轮次合并为一段连贯概要（≤200字）。"""
    try:
        prompt = _build_prompt(
            _prompts["context_merge"],
            old_summary=old_summary,
            new_turns_text=new_turns_text,
        )
        response = await model.ainvoke([HumanMessage(content=prompt)])
        return str(response.content).strip()[:300]
    except Exception:
        logger.warning("合并上下文摘要失败", exc_info=True)
        return old_summary


async def _process_session(db_config: dict, session: dict) -> int:
    """处理一个会话：为未概括的轮次生成摘要，更新全局上下文摘要。"""
    thread_id = session["thread_id"]
    turn_count = session["turn_count"]
    existing_context = session.get("context_summary") or ""

    max_summarize_index = turn_count - 4
    if max_summarize_index < 0:
        return 0

    logger.info(f"[{thread_id[:8]}] 会话有 {turn_count} 轮，需概括到第 {max_summarize_index} 轮")

    with mysql.connector.connect(**db_config) as conn:
        with conn.cursor(dictionary=True) as cur:
            cur.execute("SELECT id FROM sessions WHERE thread_id = %s", (thread_id,))
            srow = cur.fetchone()
            if not srow:
                return 0
            session_id = srow["id"]

            cur.execute(
                "SELECT id, turn_index, user_query, assistant_result, summary"
                " FROM conversations WHERE session_id = %s ORDER BY turn_index ASC",
                (session_id,),
            )
            all_turns = cur.fetchall()

        pending_turns = [
            t for t in all_turns
            if t["turn_index"] <= max_summarize_index and not (t.get("summary") or "").strip()
        ]

        processed = 0

        for turn in pending_turns:
            summary = await _generate_turn_summary(
                turn["user_query"], turn["assistant_result"] or ""
            )
            if summary:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE conversations SET summary = %s WHERE id = %s",
                        (summary, turn["id"]),
                    )
                processed += 1

        if processed > 0 or not existing_context:
            summarized_turns = [
                t for t in all_turns if t["turn_index"] <= max_summarize_index
            ]
            turns_text_parts = []
            for t in summarized_turns:
                s = (t.get("summary") or "").strip()
                result = s or (t["assistant_result"] or "")[:200]
                turns_text_parts.append(f"用户: {t['user_query']}\n助手: {result}")
            all_turns_text = "\n\n".join(turns_text_parts)

            if existing_context and processed > 0:
                new_context = await _merge_context_summary(existing_context, all_turns_text)
            else:
                new_context = await _generate_context_summary(all_turns_text)

            if new_context:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE sessions SET context_summary = %s WHERE id = %s",
                        (new_context, session_id),
                    )

    return processed


async def _run_loop() -> None:
    """主循环：启动时立即扫描一次，之后每 24 小时扫描一次。"""
    db_config = _get_db_config()
    logger.info("异步摘要 Agent 启动，扫描间隔 24h")

    while True:
        try:
            with mysql.connector.connect(**db_config) as conn:
                with conn.cursor(dictionary=True) as cur:
                    cur.execute(
                        "SELECT id, thread_id, turn_count, context_summary"
                        " FROM sessions WHERE turn_count > 3"
                    )
                    sessions = cur.fetchall()

            if sessions:
                logger.info(f"发现 {len(sessions)} 个需检查的会话")
                for session in sessions:
                    try:
                        n = await _process_session(db_config, session)
                        if n > 0:
                            logger.info(
                                f"[{session['thread_id'][:8]}] 概括完成，处理 {n} 轮"
                            )
                    except Exception:
                        logger.warning(
                            f"[{session['thread_id'][:8]}] 概括失败", exc_info=True,
                        )
        except Exception:
            logger.warning("扫描会话失败", exc_info=True)

        await asyncio.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    asyncio.run(_run_loop())
