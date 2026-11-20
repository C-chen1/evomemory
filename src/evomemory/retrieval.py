"""Retrieval over experiences: BM25 + hashed dense vectors + reciprocal rank fusion.

No model download, no vector database, no network. The dense channel uses a
hashing embedder, which is a *lexical* feature map dressed as a vector — that is
a deliberate limitation, stated plainly: with ``HashEmbedder`` the dense channel
adds fuzzy token overlap, not semantics. A real embedding model plugs in through
the same ``Embedder`` protocol when one is available.

The fusion step is RRF because it needs no score calibration between channels,
which matters when one channel is a tuned BM25 and the other is a unit-norm hash
vector.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from .models import Experience

_TOKEN_RE = re.compile(r"[a-z0-9_]+|[一-鿿]")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "be", "it", "this", "that", "as", "at", "by", "from",
}


def tokenize(text: str) -> List[str]:
    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]
    # CJK: add bigrams so that multi-character terms are separable
    cjk = [t for t in tokens if "一" <= t <= "鿿"]
    if len(cjk) > 1:
        tokens.extend("".join(pair) for pair in zip(cjk, cjk[1:]))
    return tokens


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, text: str) -> List[float]: ...


class HashEmbedder:
    """Deterministic hashed bag-of-words, L2 normalised.

    Same input, same vector, always — which is what makes offline runs
    reproducible. Not a semantic model.
    """

    def __init__(self, dim: int = 256, name: str = "hash-bow-256"):
        self.dim = dim
        self.name = name

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        counts = Counter(tokenize(text))
        for token, n in counts.items():
            h = hash_token(token)
            idx = h % self.dim
            sign = 1.0 if (h >> 16) & 1 else -1.0
            vec[idx] += sign * (1.0 + math.log(n))
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


def hash_token(token: str) -> int:
    """FNV-1a. Not Python's ``hash``, which is salted per process."""
    h = 0x811C9DC5
    for byte in token.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------


class BM25Index:
    def __init__(self, docs: Dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_ids: List[str] = []
        self.term_freqs: List[Counter] = []
        self.doc_len: List[int] = []
        self.df: Counter = Counter()
        for doc_id, text in docs.items():
            tokens = tokenize(text)
            self.doc_ids.append(doc_id)
            self.term_freqs.append(Counter(tokens))
            self.doc_len.append(len(tokens))
            for term in set(tokens):
                self.df[term] += 1
        self.n_docs = len(self.doc_ids)
        self.avg_len = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, limit: int = 20) -> List[Tuple[str, float]]:
        q_terms = tokenize(query)
        scores: List[Tuple[str, float]] = []
        for i, doc_id in enumerate(self.doc_ids):
            tf = self.term_freqs[i]
            dl = self.doc_len[i] or 1
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
                score += self._idf(term) * (f * (self.k1 + 1)) / denom
            if score > 0:
                scores.append((doc_id, score))
        # stable sort: ties keep the index's insertion order, which is itself
        # deterministic, unlike the random document ids
        scores.sort(key=lambda x: -x[1])
        return scores[:limit]

# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------


def rrf(rankings: Dict[str, List[Tuple[str, float]]], k: int = 60) -> Dict[str, float]:
    """Reciprocal rank fusion over named channels of ``(doc_id, score)``."""
    fused: Dict[str, float] = {}
    for _channel, ranking in rankings.items():
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


# --------------------------------------------------------------------------
# experience index
# --------------------------------------------------------------------------


@dataclass
class IndexedExperience:
    experience: Experience
    text: str
    vector: List[float] = field(default_factory=list)


class ExperienceIndex:
    """In-memory index over active experiences.

    Rebuilt from the store on demand — at this scale (tens to low thousands of
    experiences) an exact linear scan is both faster and easier to argue about
    than an ANN structure.
    """

    def __init__(self, experiences: Iterable[Experience],
                 embedder: Optional[Embedder] = None,
                 caption_map: Optional[Dict[str, str]] = None):
        self.embedder = embedder or HashEmbedder()
        self.entries: List[IndexedExperience] = []
        for exp in experiences:
            caption = (caption_map or {}).get(exp.experience_id, "")
            text = self._document_text(exp, caption)
            self.entries.append(IndexedExperience(exp, text, self.embedder.embed(text)))
        self.bm25 = BM25Index({e.experience.experience_id: e.text for e in self.entries}
                              ) if self.entries else None
        self._by_id = {e.experience.experience_id: e for e in self.entries}

    @staticmethod
    def _document_text(exp: Experience, caption: str) -> str:
        parts = [
            exp.lesson,
            exp.failure_signature,
            " ".join(exp.applicability_conditions),
            exp.family,
            exp.task_context,
        ]
        if caption:
            parts.append(caption)
        return "\n".join(p for p in parts if p)

    def search(self, query: str, limit: int = 5,
               family: Optional[str] = None) -> List[Tuple[Experience, float, Dict[str, float]]]:
        if not self.entries:
            return []
        ranked: Dict[str, List[Tuple[str, float]]] = {}
        if self.bm25 is not None:
            ranked["bm25"] = self.bm25.search(query, limit=limit * 4)

        qvec = self.embedder.embed(query)
        dense = sorted(
            ((e.experience.experience_id, cosine(qvec, e.vector)) for e in self.entries),
            key=lambda x: -x[1],
        )[: limit * 4]
        ranked["dense"] = dense

        fused = rrf(ranked)
        scored: List[Tuple[Experience, float, Dict[str, float]]] = []
        for exp_id, score in fused.items():
            entry = self._by_id[exp_id]
            if family and entry.experience.family and entry.experience.family != family:
                continue
            channels = {
                "bm25": dict(ranked["bm25"]).get(exp_id, 0.0),
                "dense": dict(ranked["dense"]).get(exp_id, 0.0),
            }
            scored.append((entry.experience, score, channels))
        # Tie-break on document text, not on the random experience id: sorting by
        # a uuid makes two identical runs return memories in different orders,
        # which silently turns a reproducible experiment into a lottery.
        scored.sort(key=lambda x: (-x[1], self._by_id[x[0].experience_id].text))
        return scored[:limit]

    def size(self) -> int:
        return len(self.entries)
