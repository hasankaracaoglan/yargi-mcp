# yargitay_mcp_module/karararama.py
"""Client for Yargıtay's renewed decision search (karararama.yargitay.gov.tr).

The site was rebuilt as a single-page app; the old ``POST /aramalist`` used by
``client.py`` now returns the app's HTML. The app talks to a public JSON API
(no authentication), measured 2026-11-27:

* ``GET /api/kararlar`` with ``searchText``, ``page`` (0-based), ``size``
  (up to 100), ``sortOrder=tarih_desc`` (omitted = relevance) and filters
  ``daireIds``, ``ustKararTuruIds``, ``davaTuruIds``, ``mahkemeTuruIds``,
  ``ilIds`` (repeated parameters), ``esasYilMin/Max``, ``esasSiraMin/Max``,
  ``kararYilMin/Max``, ``kararSiraMin/Max``, ``kararTarihiMin/Max``
  (YYYY-MM-DD; one-sided bounds work). At most 1,000 results per query.
* ``GET /api/kararlar/karar/{id}``: the decision HTML in ``kvtcontent``;
  it is the same text Bedesten serves (normalised comparison was identical).
* ``GET /api/{daireler,ust-karar-turleri,dava-turleri,mahkeme-turleri,iller}/all``.

Search behaviour: every word is required; short queries also match nearby
spellings and suffixed forms (``taahütname`` finds ``taahhüdü``). It is not a
semantic search: an everyday-language query returns decisions that merely
contain the words. Quotes give an exact phrase, ``-word`` excludes; ``*``,
``AND``/``OR`` are not operators (``*`` is rejected, ``OR`` is searched as a
word). Text: 3-250 characters, at most 30 words. Each hit carries
``matchType``: EXACT (the words, also misspelled or suffixed, are in the
text) or FUZZY_ONLY (only a similar word is: tahliye -> tavsiye, tahkime);
relevance order puts the FUZZY_ONLY hits last.

Coverage: about 10 million decisions from 2006 on (two before 2006); the
İÇTİHADI BİRLEŞTİRME outcome filter returns nothing.

Petitions cite Yargıtay decisions from mevzuat.adalet.gov.tr and verify
quotes against Bedesten, so each decision can be matched to its Bedesten
documentId by chamber, esas and karar number.
"""

import asyncio
import difflib
import html
import io
import logging
import os
import re
import sqlite3
import threading
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import httpx
from markitdown import MarkItDown

from bedesten_mcp_module.kunye import DATA_DIR, format_kunye

logger = logging.getLogger(__name__)

BASE_URL = "https://karararama-services.yargitay.gov.tr"
SITE_URL = "https://karararama.yargitay.gov.tr"

MAX_RESULTS = 1000      # the server refuses pages beyond the first 1,000 hits
MAX_PAGE_SIZE = 100
MIN_TEXT, MAX_TEXT, MAX_WORDS = 3, 250, 30
LOOKUP_TTL_S = 7 * 24 * 3600
NEGATIVE_MATCH_TTL_S = 7 * 24 * 3600
MAX_DAVA_TURU_IDS = 50  # the server accepts at most 50 values per filter

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_NUMBER_IN_TEXT = re.compile(r"\b\d{4}\s*/\s*\d+\b")
_NO_VALUE = re.compile(r"^\s*(\d{4})(?:\s*/\s*(\d+))?\s*$")
_TAG = re.compile(r"<[^>]+>")
_CLOSED = re.compile(r"\(\s*kapatılan\s*\)\s*", re.I)

LOOKUPS = {
    "daire": "daireler",
    "karar_turu": "ust-karar-turleri",
    "dava_turu": "dava-turleri",
    "mahkeme_turu": "mahkeme-turleri",
    "il": "iller",
}

_DAIRE_CODES = {
    "HGK": "Hukuk Genel Kurulu",
    "CGK": "Ceza Genel Kurulu",
    "BGK": "Büyük Genel Kurulu",
    "BK": "Başkanlar Kurulu",
    "HBK": "Hukuk Daireleri Başkanlar Kurulu",
    "CBK": "Ceza Daireleri Başkanlar Kurulu",
}
_DAIRE_CODE = re.compile(r"^([HC])\s*(\d{1,2})$", re.I)


class KarararamaError(Exception):
    """A request the server or the parameters refuse; the message is shown to the caller."""


def fold(text: str) -> str:
    """Case- and accent-insensitive Turkish comparison key."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("İ", "i").replace("I", "ı").lower()
    text = text.translate(str.maketrans("âîû", "aiu"))
    return re.sub(r"\s+", " ", text).strip()


def clean_daire(name: str) -> str:
    """'(Kapatılan)15. Hukuk Dairesi' -> '15. Hukuk Dairesi' (Bedesten's form)."""
    return re.sub(r"\s+", " ", _CLOSED.sub("", name or "")).strip()


def _daire_key(name: str) -> str:
    return fold(clean_daire(name)).replace(" ", "")


def _dmy(iso_date: Optional[str]) -> Optional[str]:
    if not iso_date:
        return None
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", iso_date)
    return f"{m.group(3)}.{m.group(2)}.{m.group(1)}" if m else None


def kunye_of(item: Dict[str, Any]) -> Dict[str, Optional[str]]:
    daire = clean_daire(item.get("daireAdi") or "")
    esas = f"{item['dosyaYil']}/{item['dosyaSira']}" if item.get("dosyaYil") and item.get("dosyaSira") else None
    karar = f"{item['kararYil']}/{item['kararSira']}" if item.get("kararYil") and item.get("kararSira") else None
    tarih = _dmy(item.get("kararTarihi"))
    return {
        "mahkeme": f"Yargıtay {daire}" if daire else "Yargıtay",
        "daire": daire or None,
        "esas_no": esas,
        "karar_no": karar,
        "karar_tarihi": tarih,
        "kunye": format_kunye("YARGITAYKARARI", daire, esas, karar, tarih),
    }


def plain_snippet(snippet: Optional[str], limit: int = 400) -> Optional[str]:
    if not snippet:
        return None
    text = re.sub(r"\s+", " ", html.unescape(_TAG.sub("", snippet))).strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def parse_no(value: str, label: str) -> Tuple[Optional[int], Optional[int]]:
    """'2022/8423' -> (2022, 8423); '2022' -> (2022, None)."""
    if not value or not value.strip():
        return None, None
    m = _NO_VALUE.match(value)
    if not m:
        raise KarararamaError(f"{label} '{value}' anlaşılamadı; '2022/8423' ya da yalnız yıl ('2022') yazın.")
    return int(m.group(1)), (int(m.group(2)) if m.group(2) else None)


def prepare_text(text: str) -> Tuple[str, List[str]]:
    """Adapts a query to the server's syntax; returns (text, notes)."""
    notes: List[str] = []
    text = (text or "").strip()
    if not text:
        return "", notes
    if _NUMBER_IN_TEXT.search(text):
        raise KarararamaError(
            "Arama metninde esas/karar numarası var. Numara için esas_no ve karar_no alanlarını kullanın "
            "(metinde '2022/8423' bütün kararları getirir)."
        )
    if re.search(r"\bOR\b", text):
        raise KarararamaError(
            "Bu sitede OR yok; bütün kelimeler birlikte aranır. Eş anlamlıları ayrı aramalarla deneyin."
        )
    new = re.sub(r"\bNOT\s+", "-", text)
    if new != text:
        notes.append("NOT, '-kelime' olarak gönderildi.")
        text = new
    new = re.sub(r"\bAND\b", " ", text)
    if new != text:
        notes.append("AND çıkarıldı: bütün kelimeler zaten birlikte aranır.")
        text = new
    new = re.sub(r"[*?~()+]", " ", text)
    if new != text:
        notes.append("*, ?, ~, +, parantez çıkarıldı: site ekleri ve yakın yazımları kendisi bulur.")
        text = new
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < MIN_TEXT:
        raise KarararamaError("Arama metni en az 3 karakter olmalı (ya da yalnız filtre verin).")
    if len(text) > MAX_TEXT:
        raise KarararamaError(f"Arama metni en fazla {MAX_TEXT} karakter olabilir ({len(text)} karakter).")
    if len(text.split()) > MAX_WORDS:
        raise KarararamaError(f"Arama metninde en fazla {MAX_WORDS} kelime olabilir.")
    return text, notes


class _MatchStore:
    """Yargıtay id -> Bedesten documentId (None = searched, not in Bedesten)."""

    def __init__(self, db_path: str = os.path.join(DATA_DIR, "kunye.sqlite3")):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS yargitay_bedesten ("
                " yargitay_id TEXT PRIMARY KEY, document_id TEXT, checked_at REAL)"
            )
            self._conn.commit()

    def get(self, yargitay_id: str) -> Tuple[bool, Optional[str]]:
        """(known, documentId). A miss is forgotten after NEGATIVE_MATCH_TTL_S,
        since Bedesten may publish the decision later."""
        with self._lock:
            row = self._conn.execute(
                "SELECT document_id, checked_at FROM yargitay_bedesten WHERE yargitay_id = ?",
                (yargitay_id,),
            ).fetchone()
        if row is None:
            return False, None
        if row[0] is None and time.time() - (row[1] or 0) > NEGATIVE_MATCH_TTL_S:
            return False, None
        return True, row[0]

    def put(self, yargitay_id: str, document_id: Optional[str]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO yargitay_bedesten VALUES (?, ?, ?)",
                (yargitay_id, document_id, time.time()),
            )
            self._conn.commit()


class KarararamaClient:
    """Async client for karararama-services.yargitay.gov.tr."""

    # Not throttled in tests (20 sequential searches, all 200); a short
    # spacing keeps bulk use polite.
    _MIN_INTERVAL_S = float(os.getenv("YARGITAY_MIN_INTERVAL_S", "0.3"))

    def __init__(self, request_timeout: float = 60.0):
        self.http_client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={
                "Accept": "application/json",
                "Accept-Language": "tr-TR,tr;q=0.9",
                "Origin": SITE_URL,
                "Referer": SITE_URL + "/",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
            },
            timeout=request_timeout,
        )
        self._lookups: Dict[str, Tuple[float, List[Dict[str, str]]]] = {}
        self._lookup_lock = asyncio.Lock()
        self._gap_lock = asyncio.Lock()
        self._last_request = 0.0
        self._matches: Optional[_MatchStore] = None

    # --- HTTP -------------------------------------------------------------

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        for attempt in range(3):
            async with self._gap_lock:
                wait = self._last_request + self._MIN_INTERVAL_S - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                self._last_request = time.monotonic()
            try:
                response = await self.http_client.get(path, params=params)
            except httpx.TransportError:
                if attempt < 2:
                    await asyncio.sleep(2.0)
                    continue
                raise
            if response.status_code in (429, 502, 503, 504) and attempt < 2:
                try:
                    retry_after = float(response.headers.get("Retry-After", "3"))
                except ValueError:
                    retry_after = 3.0
                await asyncio.sleep(max(1.0, min(retry_after, 30.0)))
                continue
            if response.status_code == 400:
                try:
                    message = response.json().get("message")
                except ValueError:
                    message = None
                raise KarararamaError(f"Yargıtay sitesi isteği reddetti: {message or response.text[:200]}")
            if response.status_code == 404:
                raise KarararamaError("Karar bulunamadı (HTTP 404); id'yi arama sonucundan aynen alın.")
            if response.status_code == 429:
                raise KarararamaError("Yargıtay sitesi istek sınırı (HTTP 429); biraz sonra tekrar deneyin.")
            if response.status_code >= 500:
                raise KarararamaError(f"Yargıtay sitesi şu an yanıt vermiyor (HTTP {response.status_code}); biraz sonra tekrar deneyin.")
            response.raise_for_status()
            if "json" not in response.headers.get("content-type", ""):
                raise KarararamaError(
                    "Yargıtay sitesi JSON yerine başka bir yanıt döndürdü; arka uç değişmiş olabilir."
                )
            return response.json()
        raise KarararamaError("Yargıtay sitesi istek sınırı (HTTP 429); biraz sonra tekrar deneyin.")

    # --- Filter lists -----------------------------------------------------

    async def lookup(self, kind: str) -> List[Dict[str, str]]:
        async with self._lookup_lock:
            cached = self._lookups.get(kind)
            if cached and time.time() - cached[0] < LOOKUP_TTL_S:
                return cached[1]
            data = await self._get(f"/api/{LOOKUPS[kind]}/all")
            items = [{"id": x["id"], "title": x["title"]} for x in data if x.get("id") and x.get("title")]
            self._lookups[kind] = (time.time(), items)
            return items

    async def resolve(self, kind: str, values: str) -> Tuple[List[str], List[str]]:
        """Comma-separated names -> (ids, matched titles). Raises with
        suggestions when a name matches nothing."""
        tokens = [t.strip() for t in (values or "").split(",") if t.strip()]
        if not tokens:
            return [], []
        items = await self.lookup(kind)
        ids: List[str] = []
        titles: List[str] = []
        for token in tokens:
            hits = self._match(kind, token, items)
            if not hits:
                close = difflib.get_close_matches(
                    fold(token), [fold(x["title"]) for x in items], n=5, cutoff=0.5
                )
                names = [x["title"] for x in items if fold(x["title"]) in close]
                hint = f" Yakın olanlar: {', '.join(names)}." if names else ""
                raise KarararamaError(f"{_KIND_LABEL[kind]} '{token}' bulunamadı.{hint}")
            for x in hits:
                if x["id"] not in ids:
                    ids.append(x["id"])
                    titles.append(x["title"])
        if kind == "dava_turu" and len(ids) > MAX_DAVA_TURU_IDS:
            raise KarararamaError(
                f"dava_turu {len(ids)} türe uyuyor; daha belirli yazın "
                f"(örn. 'Kiralananın Tahliyesi'). İlk birkaçı: {', '.join(titles[:5])}."
            )
        return ids, titles

    @staticmethod
    def _match(kind: str, token: str, items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if kind == "daire":
            m = _DAIRE_CODE.match(token)
            if m:
                token = f"{int(m.group(2))}. {'Hukuk' if m.group(1).upper() == 'H' else 'Ceza'} Dairesi"
            token = _DAIRE_CODES.get(token.upper(), token)
            key = _daire_key(token)
            return [x for x in items if _daire_key(x["title"]) == key]
        key = fold(token)
        exact = [x for x in items if fold(x["title"]) == key]
        if kind == "il":
            return exact
        if kind == "dava_turu":
            # One kind of case has several titles over the years: 'Kiralananın
            # Tahliyesi' (old), 'Borçlar Yasasından Kaynaklanan (Kiralananın
            # Tahliyesi)' (current), ... so every title containing the name is
            # used; an exact title only wins when the name is too general.
            hits = [x for x in items if key in fold(x["title"])]
            return exact if exact and len(hits) > MAX_DAVA_TURU_IDS else hits
        if exact:
            return exact
        return [x for x in items if key in fold(x["title"])]

    # --- Search and documents ---------------------------------------------

    async def search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return await self._get("/api/kararlar", params=params)

    async def get_document(self, yargitay_id: str) -> Dict[str, Any]:
        return await self._get(f"/api/kararlar/karar/{yargitay_id}")

    @staticmethod
    def html_to_markdown(content: str) -> str:
        return MarkItDown().convert(io.BytesIO(content.encode("utf-8"))).text_content

    # --- Bedesten match ----------------------------------------------------

    def match_store(self) -> _MatchStore:
        if self._matches is None:
            self._matches = _MatchStore()
        return self._matches

    async def bedesten_match(self, yargitay_id: str, kunye: Dict[str, Optional[str]], bedesten_client) -> Optional[str]:
        """Bedesten documentId of the same decision, or None if Bedesten has
        no decision with this chamber, esas and karar number. Rate-limit and
        network errors propagate (and are not cached as a miss)."""
        try:
            known, document_id = self.match_store().get(yargitay_id)
            if known:
                return document_id
        except Exception as e:
            logger.warning(f"KarararamaClient: match store read failed: {e}")
        if not (kunye.get("esas_no") and kunye.get("karar_no")):
            return None
        from bedesten_mcp_module.enums import BIRIM_ADI_MAPPING
        from bedesten_mcp_module.models import BedestenSearchRequest, BedestenSearchData
        from bedesten_mcp_module.sorgu import date_range

        want = _daire_key(kunye.get("daire") or "")

        async def find(**data) -> Optional[str]:
            request = BedestenSearchRequest(data=BedestenSearchData(
                itemTypeList=["YARGITAYKARARI"], pageSize=10, pageNumber=1, **data
            ))
            response = await bedesten_client.search_documents(request)
            for e in (response.data.emsalKararList if response.data else None) or []:
                if e.esasNo == kunye["esas_no"] and e.kararNo == kunye["karar_no"] and _daire_key(e.birimAdi or "") == want:
                    return e.documentId
            return None

        # Bedesten searches the numbers in the text, where old general-assembly
        # headers read '2010/15-193 E.' while the record says 2010/193; the
        # second query looks for the karar number in that chamber on that day.
        document_id = await find(phrase=f"{kunye['esas_no']} AND {kunye['karar_no']}")
        if document_id is None and kunye.get("karar_tarihi"):
            code = next((k for k, v in BIRIM_ADI_MAPPING.items()
                         if v and _daire_key(v) == want and k != "ALL"), None)
            start, end = date_range(kunye["karar_tarihi"], kunye["karar_tarihi"])
            extra = {"birimAdi": code} if code else {}
            document_id = await find(phrase=kunye["karar_no"], kararTarihiStart=start, kararTarihiEnd=end, **extra)
        try:
            self.match_store().put(yargitay_id, document_id)
        except Exception as e:
            logger.warning(f"KarararamaClient: match store write failed: {e}")
        return document_id

    async def close_client_session(self):
        await self.http_client.aclose()


_KIND_LABEL = {
    "daire": "Daire",
    "karar_turu": "Karar türü",
    "dava_turu": "Dava/suç türü",
    "mahkeme_turu": "Mahkeme türü",
    "il": "İl",
}


def site_link(yargitay_id: str) -> str:
    return f"{SITE_URL}/kararlar/{yargitay_id}"
