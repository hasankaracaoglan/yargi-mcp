# bedesten_mcp_module/kunye.py
"""Decision citation (künye) helpers and a persistent documentId -> künye store.

Bedesten's document endpoint returns only the decision text; the official
chamber / esas / karar / date exist only in search results. Every search
result is recorded here so the document tool can attach the official künye
instead of leaving the date to be guessed from the text, which also contains
lower-court and BAM dates.
"""

import logging
import os
import re
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "veri")
DEFAULT_DB_PATH = os.path.join(DATA_DIR, "kunye.sqlite3")

# For these types birimAdi has no court name ("3. Hukuk Dairesi", "4. Daire");
# KYB decisions are rendered by Yargıtay criminal chambers. BAM and local court
# names already include the court ("Kayseri Bölge Adliye Mahkemesi 6. Hukuk Dairesi").
_COURT_PREFIX = {
    "YARGITAYKARARI": "Yargıtay",
    "KYB": "Yargıtay",
    "DANISTAYKARAR": "Danıştay",
}


def court_name(item_type: Optional[str], birim_adi: Optional[str]) -> Optional[str]:
    birim = (birim_adi or "").strip()
    prefix = _COURT_PREFIX.get(item_type or "")
    if not birim:
        return prefix
    if prefix and not birim.startswith(prefix):
        return f"{prefix} {birim}"
    return birim


def format_kunye(
    item_type: Optional[str],
    birim_adi: Optional[str],
    esas_no: Optional[str],
    karar_no: Optional[str],
    karar_tarihi: Optional[str],
) -> Optional[str]:
    """Citation in the petition template's form, e.g.
    'Yargıtay 3. Hukuk Dairesi, 2022/8423 Esas, 2023/291 Karar, 23.02.2023 tarihli'."""
    parts = []
    name = court_name(item_type, birim_adi)
    if name:
        parts.append(name)
    if esas_no:
        parts.append(f"{esas_no} Esas")
    if karar_no:
        parts.append(f"{karar_no} Karar")
    if karar_tarihi:
        parts.append(f"{karar_tarihi} tarihli")
    return ", ".join(parts) if parts else None


# Decision headers: Yargıtay/Danıştay "2022/8423 E. , 2023/291 K." and BAM
# "Esas-Karar No: 2025/240 - 2026/1178" or "ESAS|DOSYA NO : ... KARAR NO : ...".
_HEADER_PATTERNS = (
    re.compile(r"(\d{4})/(\d+)\s*E\.?\s*,\s*(\d{4})/(\d+)\s*K\b"),
    re.compile(r"Esas-Karar No\s*:\s*(\d{4})/(\d+)\s*-\s*(\d{4})/(\d+)", re.IGNORECASE),
)
_ESAS_LINE = re.compile(r"(?:ESAS|DOSYA)\s*NO\s*:\s*(\d{4})/(\d+)", re.IGNORECASE)
_KARAR_LINE = re.compile(r"KARAR\s*NO\s*:\s*(\d{4})/(\d+)", re.IGNORECASE)


def numbers_from_header(text: str) -> Optional[Tuple[str, str]]:
    """(esas_no, karar_no) such as ('2022/8423', '2023/291') from the decision
    header, or None. Only the header is read: the body cites lower-court numbers."""
    head = text[:800]
    for pattern in _HEADER_PATTERNS:
        m = pattern.search(head)
        if m:
            return f"{m.group(1)}/{m.group(2)}", f"{m.group(3)}/{m.group(4)}"
    esas, karar = _ESAS_LINE.search(head), _KARAR_LINE.search(head)
    if esas and karar:
        return f"{esas.group(1)}/{esas.group(2)}", f"{karar.group(1)}/{karar.group(2)}"
    return None


class KunyeStore:
    """SQLite map of documentId -> official künye; survives server restarts."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS kunye ("
                " document_id TEXT PRIMARY KEY, item_type TEXT, birim_adi TEXT,"
                " esas_no TEXT, karar_no TEXT, karar_tarihi TEXT, updated_at REAL)"
            )
            self._conn.commit()

    def save_entries(self, entries: Iterable[Any]) -> None:
        """Record BedestenDecisionEntry objects from a search response."""
        now = time.time()
        rows = [
            (
                e.documentId,
                getattr(getattr(e, "itemType", None), "name", None),
                e.birimAdi,
                e.esasNo,
                e.kararNo,
                e.kararTarihiStr,
                now,
            )
            for e in entries
        ]
        if not rows:
            return
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO kunye VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
            self._conn.commit()

    def get(self, document_id: str) -> Optional[Dict[str, Optional[str]]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT item_type, birim_adi, esas_no, karar_no, karar_tarihi"
                " FROM kunye WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        if row is None:
            return None
        keys = ("item_type", "birim_adi", "esas_no", "karar_no", "karar_tarihi")
        return dict(zip(keys, row))


_store: Optional[KunyeStore] = None


def get_store() -> KunyeStore:
    global _store
    if _store is None:
        _store = KunyeStore()
    return _store
