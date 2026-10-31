# bedesten_mcp_module/sorgu.py
"""Query rewriting for Bedesten's search syntax.

Bedesten matches whole word forms only (no stemming: taahhütname does not
match taahhütnamesi) and returns no data for a query containing '/'.
rewrite_phrase expands `kök*` into the word's common Turkish suffix forms and
sends slashed terms such as case numbers as phrases.
"""

import re
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Tuple

_VOWELS = "aıoueiöüâîû"
_BACK = "aıouâû"
_VOICELESS = "çfhkpsşt"
_SOFTENED = {"p": "b", "ç": "c", "t": "d", "k": "ğ"}
_BACK_I = {"a": "ı", "ı": "ı", "o": "u", "u": "u", "â": "ı", "û": "u"}
_FRONT_I = {"e": "i", "i": "i", "ö": "ü", "ü": "ü", "î": "i",
            "a": "i", "ı": "i", "o": "ü", "u": "ü", "â": "i", "û": "ü"}

# Stem before a vowel-initial suffix for legal words that double their last
# consonant (hak -> hakkı) or drop their last vowel (hüküm -> hükmü, haciz -> haczi).
_VOWEL_STEMS = {
    "hak": "hakk", "ret": "redd", "zan": "zann", "af": "aff", "his": "hiss", "sır": "sırr",
    "akit": "akd", "ahit": "ahd", "nakit": "nakd", "vakit": "vakt", "kasıt": "kast",
    "kayıt": "kayd", "zabıt": "zapt", "hüküm": "hükm", "cürüm": "cürm", "fesih": "fesh",
    "haciz": "hacz", "rehin": "rehn", "mehir": "mehr", "gabin": "gabn", "cebir": "cebr",
    "hapis": "haps", "keşif": "keşf", "devir": "devr", "nakil": "nakl", "şekil": "şekl",
    "metin": "metn", "emir": "emr", "isim": "ism", "fikir": "fikr", "asıl": "asl",
    "akıl": "akl", "kısım": "kısm", "resim": "resm", "hasım": "hasm", "vasıf": "vasf",
    "ömür": "ömr", "şehir": "şehr", "seyir": "seyr", "nesil": "nesl", "sabır": "sabr",
    "ağız": "ağz", "zikir": "zikr", "nehir": "nehr",
}

# Back-vowel words that take front-vowel suffixes (dikkat -> dikkati, beraat ->
# beraati). Back-vowel words ending in l are tried both ways (usul -> usulü).
_FRONT_EXCEPTIONS = {
    "saat", "dikkat", "kanaat", "itaat", "cemaat", "hakikat", "beraat", "kabahat",
    "sıhhat", "harf", "kalp",
}


def suffix_forms(word: str) -> List[str]:
    """The word and its common case, possessive and plural forms.

    Wrong guesses (both softened and unsoftened stems, both harmonies for
    exceptions) are kept on purpose: a form that does not exist matches
    nothing, a missing form loses decisions.
    """
    root = word.replace("I", "ı").replace("İ", "i").lower()
    vowels = [c for c in root if c in _VOWELS]
    if not vowels:
        return [root]
    last = vowels[-1]
    ends_with_vowel = root[-1] in _VOWELS
    if root in _VOWEL_STEMS:
        stems = [_VOWEL_STEMS[root]]
    elif root[-1] in _SOFTENED:
        # Both: taahhüt -> taahhüdü softens, tespit -> tespiti does not.
        stems = [root, root[:-1] + _SOFTENED[root[-1]]]
    else:
        stems = [root]
    harmonies = [last in _BACK]
    if harmonies[0] and (root.endswith("l") or root in _FRONT_EXCEPTIONS):
        harmonies.append(False)
    d = "t" if root[-1] in _VOICELESS else "d"

    forms = {root}
    for back in harmonies:
        a = "a" if back else "e"
        i = (_BACK_I if back else _FRONT_I)[last]
        lar, li = ("lar", "ı") if back else ("ler", "i")
        if ends_with_vowel:
            # Buffer consonants: y (accusative, dative, instrumental), n (genitive), s (possessive)
            forms.update({root + "y" + i, root + "y" + a, root + "n" + i + "n", root + "yl" + a})
            possessives = [root + "s" + i]
        else:
            forms.update(stem + a for stem in stems)
            forms.update(stem + i + "n" for stem in stems)
            forms.add(root + "l" + a)
            possessives = [stem + i for stem in stems]  # also the accusative
        forms.update(root + d + a + end for end in ("", "n", "ki"))
        for p in possessives:
            forms.update(p + end for end in (
                "", "n" + a, "n" + i, "n" + i + "n", "nd" + a, "nd" + a + "n", "nd" + a + "ki", "yl" + a,
            ))
        forms.update(root + lar + end for end in (
            "", li, a, "d" + a, "d" + a + "n", li + "n", li + "n" + a, li + "n" + li,
            li + "n" + li + "n", li + "nd" + a, li + "nd" + a + "n", li + "yl" + a,
        ))
    return sorted(forms, key=lambda f: (len(f), f))


_QUOTED = re.compile(r'"[^"]*"')
_STARRED = re.compile(r"(\w+)\*")
_SLASHED = re.compile(r"\w+(?:/\w+)+")


def rewrite_phrase(phrase: str) -> Tuple[str, List[str]]:
    """Return the phrase to send to Bedesten and notes on what was rewritten.

    Idempotent: a rewritten phrase contains no '*' or '/' outside quotes, so
    rewriting it again changes nothing.
    """
    notes: List[str] = []

    def expand(m: "re.Match[str]") -> str:
        forms = suffix_forms(m.group(1))
        if len(forms) == 1:
            return forms[0]
        notes.append(f"{m.group(0)} → {len(forms)} ek biçimiyle arandı")
        return "(" + " OR ".join(forms) + ")"

    def quote(m: "re.Match[str]") -> str:
        quoted = '"' + m.group(0).replace("/", " ") + '"'
        notes.append(f"{m.group(0)} → {quoted}")
        return quoted

    def unquoted(segment: str) -> str:
        return _SLASHED.sub(quote, _STARRED.sub(expand, segment))

    out, pos = [], 0
    for m in _QUOTED.finditer(phrase):
        out.append(unquoted(phrase[pos:m.start()]))
        quoted = m.group(0)
        fixed = quoted.replace("/", " ").replace("*", "")
        if fixed != quoted:
            notes.append(f"{quoted} → {fixed} (tırnak içinde / ve * kullanılamaz)")
        out.append(fixed)
        pos = m.end()
    out.append(unquoted(phrase[pos:]))
    return "".join(out), notes


_TURKEY = timezone(timedelta(hours=3))
_EARLIEST = date(1900, 1, 1)
_DMY = re.compile(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})$")


def date_range(start: str, end: str) -> Tuple[Optional[str], Optional[str]]:
    """Bedesten bounds for a decision-date filter given as YYYY-MM-DD or DD.MM.YYYY.

    Bedesten ignores a one-sided range (3. HD, measured 2026-10-31: start only
    48 hits = no filter; start and end 16), so a missing bound is filled with
    the earliest date or today. A day D is sent as [D-1 21:00Z, D 20:59:59.999Z]:
    the filter matched decisions on their UTC-midnight date in tests, while
    kararTarihi shows Turkish midnight in UTC (05.05.2026 -> 2026-05-04T21:00Z);
    this window holds both forms of D and neither form of its neighbours.
    Values that already contain 'T' are passed through unchanged.
    """
    if not start and not end:
        return None, None
    lo = start if "T" in start else _day_start(_parse_day(start) if start else _EARLIEST)
    hi = end if "T" in end else _day_end(_parse_day(end) if end else datetime.now(_TURKEY).date())
    return lo, hi


def _parse_day(value: str) -> date:
    value = value.strip()
    m = _DMY.match(value)
    if m:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"Tarih YYYY-MM-DD biçiminde olmalı (örn. 2020-01-01): {value!r}") from None


def _day_start(day: date) -> str:
    return f"{(day - timedelta(days=1)).isoformat()}T21:00:00.000Z"


def _day_end(day: date) -> str:
    return f"{day.isoformat()}T20:59:59.999Z"
