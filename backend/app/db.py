from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_sqlite_path(database_url: str) -> str:
    """Convert sqlite:/// or sqlite://// URLs into a filesystem path (or :memory:)."""
    if database_url in (":memory:", "sqlite:///:memory:", "sqlite://"):
        return ":memory:"
    if database_url.startswith("sqlite:////"):
        # 4 slashes => absolute path (e.g. sqlite:////data/app.db -> /data/app.db)
        return "/" + database_url[len("sqlite:////") :]
    if database_url.startswith("sqlite:///"):
        return database_url[len("sqlite:///") :]
    return database_url


class Database:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.db_path = parse_sqlite_path(database_url)
        self._memory_conn: sqlite3.Connection | None = None

        if self.db_path != ":memory:":
            parent = Path(self.db_path).parent
            if str(parent) and str(parent) != ".":
                os.makedirs(parent, exist_ok=True)
        else:
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row

        self.init_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._memory_conn is not None:
            return self._memory_conn
        conn = sqlite3.connect(self.db_path, timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _close(self, conn: sqlite3.Connection) -> None:
        if conn is not self._memory_conn:
            conn.close()

    def init_schema(self) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        system_prompt TEXT,
                        created_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata TEXT,
                        created_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_messages_session_id
                    ON messages(session_id, id);
                    """
                )
        finally:
            self._close(conn)

    def ping(self) -> bool:
        conn = self._connect()
        try:
            row = conn.execute("SELECT 1").fetchone()
            return row is not None and row[0] == 1
        except sqlite3.Error:
            return False
        finally:
            self._close(conn)

    def get_session_system_prompt(self, session_id: str) -> str | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT system_prompt FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row["system_prompt"] if row is not None else None
        finally:
            self._close(conn)

    def get_history_for_llm(self, session_id: str) -> list[dict[str, str]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            return [{"role": row["role"], "content": row["content"]} for row in rows]
        finally:
            self._close(conn)

    def save_turn(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        assistant_metadata: dict[str, Any],
        system_prompt: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        try:
            with conn:
                if system_prompt is not None:
                    conn.execute(
                        """
                        INSERT INTO sessions (session_id, system_prompt, created_at)
                        VALUES (?, ?, ?)
                        ON CONFLICT(session_id) DO UPDATE SET system_prompt = excluded.system_prompt
                        """,
                        (session_id, system_prompt, now),
                    )
                else:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO sessions (session_id, system_prompt, created_at)
                        VALUES (?, NULL, ?)
                        """,
                        (session_id, now),
                    )

                conn.execute(
                    """
                    INSERT INTO messages (session_id, role, content, metadata, created_at)
                    VALUES (?, 'user', ?, NULL, ?)
                    """,
                    (session_id, user_content, now),
                )
                conn.execute(
                    """
                    INSERT INTO messages (session_id, role, content, metadata, created_at)
                    VALUES (?, 'assistant', ?, ?, ?)
                    """,
                    (session_id, assistant_content, json.dumps(assistant_metadata), now),
                )
        finally:
            self._close(conn)

    def get_session_messages(self, session_id: str) -> list[dict[str, Any]] | None:
        conn = self._connect()
        try:
            session_row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            rows = conn.execute(
                "SELECT role, content, metadata FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()

            if session_row is None and not rows:
                return None

            result: list[dict[str, Any]] = []
            for row in rows:
                item: dict[str, Any] = {
                    "role": row["role"],
                    "content": row["content"],
                }
                if row["role"] == "assistant" and row["metadata"]:
                    item["metadata"] = json.loads(row["metadata"])
                result.append(item)
            return result
        finally:
            self._close(conn)
