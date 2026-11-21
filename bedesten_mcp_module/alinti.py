# bedesten_mcp_module/alinti.py
"""Word-for-word check of a petition quotation against a decision text.

The petition rules (dilekçe Kural 2) require every word inside quotation marks
to match the decision exactly; '...' may only mark an omitted part. A petition
quotes a decision as

    >>> *"... metin **vurgulanan kısım** metin ..."*

The check ignores whitespace and the petition's own markup (the >>> marker,
asterisks, the outer quotation marks) and nothing else: every word and
punctuation mark between the omissions must appear in the decision, in order.
A second, looser pass recognises quotes that differ only in the glyphs of
quotation marks, apostrophes or dashes, so they can be copied as the source
writes them.
"""

import difflib
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Tuple

TAM_ESLESME = "TAM_ESLESME"
BICIM_FARKI = "BICIM_FARKI"
ESLESMEDI = "ESLESMEDI"
HATA = "HATA"

_WS = re.compile(r"\s+")
_INVISIBLE = re.compile("[­​‌‍﻿]")
# Markdown escapes added by the HTML->Markdown conversion (e.g. "\-", "\.").
_MD_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>~<])")
# An omission in the quote: three or more dots, or the ellipsis character.
# Decision texts also use '...' for anonymised names; treating those as an
# omission as well only makes the check look for the parts around them.
_OMISSION = re.compile(r"\.{3,}|…")
_BLOCK_MARKER = re.compile(r"(?m)^\s*>+\s?")
# One character to one character, so positions in the loose text are
# positions in the strict text as well.
_LOOSE = str.maketrans({
    "’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'",
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "–": "-", "—": "-", "‑": "-", "−": "-",
})


def _strict(text: str) -> str:
    """Decision or quote text with formatting removed but every word and
    punctuation mark kept."""
    text = unicodedata.normalize("NFC", text)
    text = _INVISIBLE.sub("", text)
    text = _MD_ESCAPE.sub(r"\1", text)
    text = text.replace("*", "")
    return _WS.sub(" ", text).strip()


def _loose(strict_text: str) -> str:
    return strict_text.translate(_LOOSE)


def _strip_petition_markup(quote: str) -> str:
    """Removes the >>> marker, the italics around the quote and its outer
    quotation marks."""
    q = _BLOCK_MARKER.sub("", quote.strip())
    q = q.strip().strip("*").strip()
    if len(q) >= 2 and q[0] in "\"“" and q[-1] in "\"”":
        q = q[1:-1]
    return q.strip()


def _segments(quote_strict: str) -> List[str]:
    return [s.strip() for s in _OMISSION.split(quote_strict) if s.strip()]


def _find_in_order(source: str, whole: str, parts: List[str]) -> Tuple[List[Tuple[int, int]], Optional[int]]:
    """Spans of the quote in the source, or the index of the first part that
    is missing. The whole quote is tried first, so anonymisation dots that the
    quote copies exactly need no special handling."""
    at = source.find(whole)
    if at >= 0:
        return [(at, at + len(whole))], None
    spans, pos = [], 0
    for i, part in enumerate(parts):
        at = source.find(part, pos)
        if at < 0:
            return spans, i
        spans.append((at, at + len(part)))
        pos = at + len(part)
    return spans, None


def _closest_passage(source_strict: str, part_strict: str, context: int = 4) -> Dict:
    """The passage of the source most like the missing part, with the words
    that differ."""
    src = source_strict.split(" ")
    seg = part_strict.split(" ")

    def key(word: str) -> str:
        return word.translate(_LOOSE).casefold()

    src_k = [key(w) for w in src]
    seg_k = [key(w) for w in seg]
    k = len(seg_k)
    if not src or not k:
        return {}

    # Sliding window of k words that shares the most words with the part.
    need = Counter(seg_k)
    best_i = 0
    if len(src_k) > k:
        window = Counter(src_k[:k])
        score = sum(min(c, need[w]) for w, c in window.items())
        best = score
        for i in range(1, len(src_k) - k + 1):
            out_w, in_w = src_k[i - 1], src_k[i + k - 1]
            if window[out_w] <= need[out_w]:
                score -= 1
            window[out_w] -= 1
            window[in_w] += 1
            if window[in_w] <= need[in_w]:
                score += 1
            if score > best:
                best, best_i = score, i

    lo = max(0, best_i - context)
    hi = min(len(src), best_i + k + context)
    matcher = difflib.SequenceMatcher(None, seg_k, src_k[lo:hi], autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size]
    if not blocks:
        return {"en_yakin_kaynak_metni": " ".join(src[best_i:best_i + k]), "benzerlik": 0.0, "farklar": []}
    # From the first to the last shared word, widened by the part's unmatched
    # words at either end so their counterparts in the source show as well.
    first, last = blocks[0], blocks[-1]
    start = max(0, lo + first.b - first.a)
    end = min(len(src), lo + last.b + last.size + (k - last.a - last.size))
    passage = src[start:end]

    word_matcher = difflib.SequenceMatcher(None, seg, passage, autojunk=False)
    differences = []
    for tag, i1, i2, j1, j2 in word_matcher.get_opcodes():
        if tag != "equal":
            differences.append({
                "alintida": " ".join(seg[i1:i2]),
                "kaynakta": " ".join(passage[j1:j2]),
            })
    return {
        "en_yakin_kaynak_metni": " ".join(passage),
        # Share of words in common, 0-1; a low value means the passage is not
        # the same statement and the quote has no counterpart in the decision.
        "benzerlik": round(word_matcher.ratio(), 2),
        "farklar": differences,
    }


def check_quote(source_text: str, quote: str) -> Dict:
    """Compares a petition quotation with the decision text.

    durum:
      TAM_ESLESME  every word and mark appears in the decision, in order.
      BICIM_FARKI  same words; quotation mark, apostrophe or dash glyphs
                   differ. kaynaktaki_parcalar holds the source's own form.
      ESLESMEDI    a part does not appear in the decision; the closest
                   passage and the differing words are returned.
      HATA         nothing to compare.
    """
    if not source_text or not source_text.strip():
        return {"durum": HATA, "aciklama": "Karar metni boş."}
    quote_strict = _strict(_strip_petition_markup(quote))
    parts = _segments(quote_strict)
    if not parts:
        return {"durum": HATA, "aciklama": "Alıntı boş."}

    source_strict = _strict(source_text)
    spans, missing = _find_in_order(source_strict, quote_strict, parts)
    if missing is None:
        return {
            "durum": TAM_ESLESME,
            "aciklama": "Alıntı karar metninde kelimesi kelimesine geçiyor.",
            "parca_sayisi": len(parts),
        }

    source_loose = _loose(source_strict)
    quote_loose = _loose(quote_strict)
    parts_loose = _segments(quote_loose)
    spans, missing = _find_in_order(source_loose, quote_loose, parts_loose)
    if missing is None:
        # The loose text keeps the strict text's positions (1:1 mapping).
        return {
            "durum": BICIM_FARKI,
            "aciklama": (
                "Kelimeler aynı; tırnak, kesme işareti veya tire biçimi kaynaktan farklı. "
                "Alıntıyı kaynaktaki_parcalar'daki biçimle yazın."
            ),
            "parca_sayisi": len(parts),
            "kaynaktaki_parcalar": [source_strict[a:b] for a, b in spans],
        }

    # The loose pass splits the quote at the same places, so parts[missing]
    # is the strict form of the part that failed.
    if missing and source_loose.find(parts_loose[missing]) >= 0:
        problem = (
            f"Alıntının {missing + 1}. parçası karar metninde var ama önceki parçadan önce geçiyor; "
            "parçaların sırası kaynaktakiyle aynı olmalı."
        )
    else:
        problem = (
            f"Alıntının {len(parts)} parçasından {missing + 1}. parça karar metninde bulunamadı. "
            "Alıntıyı en_yakin_kaynak_metni'ndeki hâliyle düzeltin; atlanan kısım varsa yerine "
            "'...' koyun. Yakın metin aynı şeyi söylemiyorsa (benzerlik düşükse) ya da kaynakta "
            "karşılığı yoksa alıntıyı tırnaksız anlatıma çevirin."
        )
    result = {
        "durum": ESLESMEDI,
        "aciklama": problem,
        "parca_sayisi": len(parts),
        "eslesmeyen_parca": parts[missing],
    }
    result.update(_closest_passage(source_strict, parts[missing]))
    return result
