"""Keyword search for the education library: tokenizer, BM25 and rank fusion."""
import math
import re
from collections import Counter
from collections.abc import Collection, Sequence

_TOKEN = re.compile(r"\d+(?:\.\d+)?|[a-z]+")
_NUMBER = re.compile(r"\d")
_UNITS = frozenset("mg mcg g kg ml l mmol meq iu units unit oz lb mmhg".split())
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have i if in is it its me my of on "
    "or our so that the their them then there these they this to was we were "
    "what when where which who will with you your".split())


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without stopwords. "0.6 mg" also yields "0.6mg"."""
    raw = _TOKEN.findall(text.lower())
    out: list[str] = []
    for i, tok in enumerate(raw):
        if tok in _STOPWORDS:
            continue
        out.append(tok)
        if tok in _UNITS and i > 0 and _NUMBER.match(raw[i - 1]):
            out.append(raw[i - 1] + tok)
    return out


class BM25:
    """Okapi BM25 over a fixed corpus of chunk id -> text."""

    def __init__(self, docs: dict[int, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._tf = {i: Counter(tokenize(t)) for i, t in docs.items()}
        self._len = {i: sum(tf.values()) for i, tf in self._tf.items()}
        self._avg = sum(self._len.values()) / len(docs) if docs else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf.values():
            df.update(tf.keys())
        n = len(docs)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str, ids: Collection[int] | None = None) -> dict[int, float]:
        """Score documents (all, or only `ids`) against the query; only scores > 0."""
        terms = set(tokenize(query)) & self._idf.keys()
        if not terms:
            return {}
        out: dict[int, float] = {}
        for i in (self._tf if ids is None else ids):
            tf = self._tf.get(i)
            if tf is None:
                continue
            norm = self.k1 * (1 - self.b + self.b * self._len[i] / self._avg) if self._avg else self.k1
            score = sum(self._idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm)
                        for t in terms if t in tf)
            if score > 0:
                out[i] = score
        return out


def rrf(*rankings: Sequence[int], k: int = 60) -> list[int]:
    """Reciprocal rank fusion: ids best first, no duplicates, ties by first appearance."""
    total: dict[int, float] = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, start=1):
            total[i] = total.get(i, 0.0) + 1.0 / (k + rank)
    return sorted(total, key=total.__getitem__, reverse=True)
