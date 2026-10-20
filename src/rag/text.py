"""French text normalisation and tokenisation for BM25 and lexical scoring."""
from __future__ import annotations

import re
import unicodedata

FRENCH_STOPWORDS = set("""
a à au aux avec ce ces cet cette dans de des du elle elles en et eux il ils je la le les leur leurs
lui ma mais me même mes moi mon ne nos notre nous on ou où par pas pour qu que qui sa se ses son sur
ta te tes toi ton tu un une vos votre vous y d l j m n s t c qu est sont été être avoir ai as avons
avez ont a-t-il faut peut doit dois quel quelle quels quelles comment combien quoi
""".split())


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def normalize(s: str) -> str:
    s = strip_accents(s.lower())
    s = s.replace("’", "'")
    return re.sub(r"\s+", " ", s).strip()


def light_stem(token: str) -> str:
    """Very light stemming (plurals only): enough for BM25 on a small corpus."""
    if len(token) > 4 and token.endswith(("aux",)):
        return token[:-3] + "al"
    if len(token) > 3 and token.endswith(("s", "x")):
        return token[:-1]
    return token


def tokenize(s: str, stem: bool = True) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", normalize(s))
    tokens = [t for t in tokens if t not in FRENCH_STOPWORDS and (len(t) > 1 or t.isdigit())]
    return [light_stem(t) for t in tokens] if stem else tokens
