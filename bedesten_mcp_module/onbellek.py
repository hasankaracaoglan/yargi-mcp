# bedesten_mcp_module/onbellek.py
"""Persistent cache of decision texts (documentId -> Markdown).

Published decisions do not change, and every Bedesten request counts against
a limit of about one request per 3.5 s. A text is downloaded once; reading it
again, or re-running the same research later, costs no request.
"""

import os
import sqlite3
import threading
import time
from typing import Optional, Tuple

from .kunye import DATA_DIR

DEFAULT_DB_PATH = os.path.join(DATA_DIR, "karar_metinleri.sqlite3")


class TextCache:
    """SQLite map of documentId -> (markdown, mime_type)."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS metin ("
                " document_id TEXT PRIMARY KEY, mime_type TEXT, markdown TEXT NOT NULL, saved_at REAL)"
            )
            self._conn.commit()

    def get(self, document_id: str) -> Optional[Tuple[str, Optional[str]]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT markdown, mime_type FROM metin WHERE document_id = ?", (document_id,)
            ).fetchone()
        return (row[0], row[1]) if row else None

    def put(self, document_id: str, markdown: str, mime_type: Optional[str]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO metin VALUES (?, ?, ?, ?)",
                (document_id, mime_type, markdown, time.time()),
            )
            self._conn.commit()

    def __contains__(self, document_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM metin WHERE document_id = ?", (document_id,)
            ).fetchone()
        return row is not None


_cache: Optional[TextCache] = None


def get_text_cache() -> TextCache:
    global _cache
    if _cache is None:
        _cache = TextCache()
    return _cache
