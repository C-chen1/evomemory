import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture()
def tmp_env(tmp_path):
    """A self-contained (db, evidence store) pair."""
    from evomemory.db import Database
    from evomemory.evidence import EvidenceStore

    db = Database(tmp_path / "memory.db")
    evidence = EvidenceStore(tmp_path / "evidence", db)
    yield db, evidence
    db.close()


@pytest.fixture()
def store(tmp_env):
    from evomemory.memory import ExperienceStore

    db, _ = tmp_env
    return ExperienceStore(db)
