"""Text embedding adapters for the knowledge base.

Three implementations, and the difference between them is not cosmetic:

* ``FastEmbedEmbedder`` — a real neural text embedding model (BAAI/bge-small-zh-v1.5,
  512-d, ONNX). This is what "semantic retrieval" means, and only results produced
  by it may be described as semantic.
* ``HashEmbedder`` — the deterministic feature-hashing vector already used for
  experience retrieval. Same text, same vector, but it is a *lexical* feature map:
  it cannot match a paraphrase that shares no tokens. Any run using it is labelled
  ``lexical`` and is not reported as semantic retrieval.
* ``StubEmbedder`` — for tests: a fixed, hand-written vector per known phrase.

The model runs locally (no API, no per-call cost), and its files are cached inside
the project directory.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence

from .retrieval import HashEmbedder, tokenize


class Embedder(Protocol):
    name: str
    dim: int

    @property
    def kind(self) -> str:
        """``semantic`` for a neural model, ``lexical`` for feature hashing."""

    def embed(self, texts: Sequence[str]) -> List[List[float]]: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# --------------------------------------------------------------------------
# real model
# --------------------------------------------------------------------------


@dataclass
class FastEmbedEmbedder:
    """A genuine text embedding model, run locally through fastembed (ONNX).

    Optional at the dependency level: if ``fastembed`` is not installed this class
    raises a clear error instead of pretending to embed. Nothing in the codebase
    silently falls back to feature hashing.
    """

    #: Chinese small model (~95 MB). Swap to BAAI/bge-small-en-v1.5 for an
    #: English corpus; the class is model-agnostic.
    model_name: str = "BAAI/bge-small-zh-v1.5"
    cache_dir: Optional[str] = None
    _model: object = None
    _dim: int = 0

    @property
    def name(self) -> str:
        return f"fastembed:{self.model_name}"

    @property
    def kind(self) -> str:
        return "semantic"

    @property
    def dim(self) -> int:
        if self._dim:
            return self._dim
        self._ensure()
        return self._dim

    def _ensure(self) -> None:
        if self._model is not None:
            return
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "fastembed is not installed; install it with "
                "`pip install 'evomemory[rag]'`, or run with --embedder hash and "
                "label the results as lexical"
            ) from exc
        self._model = TextEmbedding(model_name=self.model_name,
                                    cache_dir=self.cache_dir)
        probe = next(iter(self._model.embed(["dimension probe"])))
        self._dim = len(list(probe))

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        self._ensure()
        return [list(map(float, v)) for v in self._model.embed(list(texts))]


# --------------------------------------------------------------------------
# lexical baseline (explicitly not semantic)
# --------------------------------------------------------------------------


class LexicalHashEmbedder:
    """Wraps the existing feature-hashing vector and labels it honestly."""

    kind = "lexical"

    def __init__(self):
        self._inner = HashEmbedder()

    @property
    def name(self) -> str:
        return "hash-bow-256(lexical)"

    @property
    def dim(self) -> int:
        return self._inner.dim

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._inner.embed(t) for t in texts]


# --------------------------------------------------------------------------
# test double
# --------------------------------------------------------------------------


class StubEmbedder:
    """Deterministic vectors for tests: one axis per known keyword.

    Not a model. It exists so the retrieval plumbing can be tested without
    downloading anything, and it is never reported as a semantic result.
    """

    kind = "stub"

    def __init__(self, keywords: Sequence[str]):
        self.keywords = list(keywords)
        self._dim = max(1, len(self.keywords))

    @property
    def name(self) -> str:
        return "stub-keyword"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        out = []
        for text in texts:
            tokens = set(tokenize(text))
            vec = [1.0 if kw in tokens else 0.0 for kw in self.keywords]
            if not any(vec):
                # keep a tiny constant so cosine stays defined
                vec[0] = 1e-6
            out.append(vec)
        return out


def build_embedder(kind: str, cache_dir: Optional[Path] = None) -> Embedder:
    if kind == "fastembed":
        return FastEmbedEmbedder(cache_dir=str(cache_dir) if cache_dir else None)
    if kind == "hash":
        return LexicalHashEmbedder()
    raise ValueError(f"unknown embedder {kind!r}; use 'fastembed' or 'hash'")


def vector_digest(vector: Sequence[float]) -> str:
    """Short content hash of a vector, for cache invalidation checks."""
    raw = ",".join(f"{v:.6f}" for v in vector).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]
