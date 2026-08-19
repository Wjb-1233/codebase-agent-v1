"""Agent 对话使用的持久化会话记忆。

这个 store 刻意保持轻量：按 session_id 保存历史对话轮次，让 /agent/run
可以延续上一轮调试上下文，不要求客户端每次重传完整历史；它不做用户画像。
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

from codebase_agent.agent.memory import ConversationTurn


DEFAULT_AGENT_MEMORY_DB = "agent_memory.db"

# 已执行过建表 DDL 的数据库路径（解析后的绝对路径），跨实例复用
_initialized_paths: set[str] = set()


class AgentMemoryStore:
    """基于 SQLite 的 Agent 会话记忆。"""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or os.getenv("AGENT_MEMORY_DB_PATH", DEFAULT_AGENT_MEMORY_DB)

    def init(self) -> None:
        path = Path(self.db_path).resolve()
        key = str(path)
        if key in _initialized_paths:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_memory_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_memory_session_id
                ON agent_memory_turns(session_id, id)
                """
            )
            conn.commit()
        _initialized_paths.add(key)

    def list_turns(self, session_id: str, *, limit: int = 20) -> list[ConversationTurn]:
        session_id = _validate_session_id(session_id)
        if limit <= 0:
            return []
        self.init()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM agent_memory_turns
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        rows.reverse()
        return [ConversationTurn(role=str(role), content=str(content)) for role, content in rows]

    def append_turns(self, session_id: str, turns: list[ConversationTurn]) -> None:
        session_id = _validate_session_id(session_id)
        clean_turns = [turn for turn in turns if turn.content.strip()]
        if not clean_turns:
            return
        self.init()
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO agent_memory_turns(session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                [(session_id, turn.role, turn.content, now) for turn in clean_turns],
            )
            conn.commit()


def _validate_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id 必须是非空字符串")
    value = session_id.strip()
    if len(value) > 128:
        raise ValueError("session_id 长度不能超过 128 个字符")
    return value
