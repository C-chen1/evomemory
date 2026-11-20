"""Content-addressed evidence storage.

The one rule this module exists to enforce: **an evidence file is never
rewritten**. Its path is derived from the hash of its bytes, so if the bytes
change it is a different evidence record. Only metadata may be attached.

This is a direct response to two failure modes found in the reference repos:

* CMA-Harness' ``align_image_to_model`` resizes an image and saves it back over
  the original path, destroying the evidence that a later claim points at.
* Recuris' embedding cache is keyed on document *ids* rather than document
  *bytes*, so editing a card silently keeps a stale vector.

Derived representations (thumbnails, captions) are stored as their own evidence
rows with a ``derived_from`` pointer, so the original is always recoverable.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .db import Database, dumps, loads
from .models import Evidence, EvidenceKind, now_iso


class EvidenceStore:
    def __init__(self, root: str | Path, db: Database):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = db

    # -- writing --------------------------------------------------------
    def _path_for(self, digest: str) -> Path:
        return self.root / digest[:2] / digest[2:]

    def put(
        self,
        data: bytes,
        kind: EvidenceKind,
        mime: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        derived_from: Optional[str] = None,
    ) -> Evidence:
        digest = hashlib.sha256(data).hexdigest()
        # Reuse an existing row for identical bytes of the same kind. Metadata is
        # not merged: the first writer wins, which keeps the record immutable.
        existing = self.db.query_one(
            "SELECT * FROM evidence WHERE sha256 = ? AND kind = ?", (digest, kind.value)
        )
        if existing is not None:
            return self._row_to_evidence(existing)

        path = self._path_for(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)

        ev = Evidence(
            kind=kind,
            sha256=digest,
            path=str(path),
            mime=mime or mimetypes.guess_type(str(path))[0] or "",
            bytes_len=len(data),
            metadata=metadata or {},
            derived_from=derived_from,
        )
        self.db.insert(
            "evidence",
            {
                "evidence_id": ev.evidence_id,
                "kind": ev.kind.value,
                "sha256": ev.sha256,
                "path": ev.path,
                "mime": ev.mime,
                "bytes_len": ev.bytes_len,
                "metadata_json": dumps(ev.metadata),
                "derived_from": ev.derived_from,
                "created_at": ev.created_at,
            },
        )
        return ev

    def put_text(self, text: str, kind: EvidenceKind = EvidenceKind.TEXT,
                 metadata: Optional[Dict[str, Any]] = None,
                 derived_from: Optional[str] = None) -> Evidence:
        return self.put(text.encode("utf-8"), kind,
                        mime="text/plain; charset=utf-8",
                        metadata=metadata, derived_from=derived_from)

    def put_json(self, obj: Any, kind: EvidenceKind = EvidenceKind.ENV_STATE,
                 metadata: Optional[Dict[str, Any]] = None,
                 derived_from: Optional[str] = None) -> Evidence:
        return self.put(dumps(obj).encode("utf-8"), kind,
                        mime="application/json",
                        metadata=metadata, derived_from=derived_from)

    # -- reading --------------------------------------------------------
    def get(self, evidence_id: str) -> Evidence:
        row = self.db.query_one("SELECT * FROM evidence WHERE evidence_id = ?", (evidence_id,))
        if row is None:
            raise KeyError(f"unknown evidence {evidence_id}")
        return self._row_to_evidence(row)

    def read_bytes(self, evidence_id: str) -> bytes:
        return Path(self.get(evidence_id).path).read_bytes()

    def read_text(self, evidence_id: str) -> str:
        return self.read_bytes(evidence_id).decode("utf-8", errors="replace")

    def exists(self, evidence_id: str) -> bool:
        row = self.db.query_one("SELECT 1 FROM evidence WHERE evidence_id = ?", (evidence_id,))
        return row is not None

    def children(self, evidence_id: str) -> List[Evidence]:
        rows = self.db.query("SELECT * FROM evidence WHERE derived_from = ?", (evidence_id,))
        return [self._row_to_evidence(r) for r in rows]

    def count(self) -> int:
        row = self.db.query_one("SELECT COUNT(*) AS n FROM evidence")
        return int(row["n"]) if row else 0

    @staticmethod
    def _row_to_evidence(row: Any) -> Evidence:
        return Evidence(
            evidence_id=row["evidence_id"],
            kind=EvidenceKind(row["kind"]),
            sha256=row["sha256"],
            path=row["path"],
            mime=row["mime"],
            bytes_len=int(row["bytes_len"]),
            metadata=loads(row["metadata_json"], {}),
            derived_from=row["derived_from"],
            created_at=row["created_at"],
        )
