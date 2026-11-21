"""Resume consistency: scoring and learning are two stages, not one.

A task's outcome row is written before its experience extraction runs. If the
process dies in between, the task used to look complete on resume and its lesson
was silently lost. These tests pin the recovery behaviour.
"""

import sqlite3

import pytest

from evomemory.config import Settings
from evomemory.models import MemoryMode
from evomemory.runner import run_experiment


def _run(tmp_path, **kw):
    settings = Settings.from_env(root=tmp_path)
    settings.api_key = ""
    return run_experiment(settings, [MemoryMode.B_VECTOR_MEMORY], "scripted",
                          per_family=1, families=("escalation",),
                          out_root=tmp_path / "out", **kw)


def _db_path(tmp_path):
    return next((tmp_path / "out").glob("run_*/mode_B/memory.db"))


def test_normal_run_marks_memory_settled(tmp_path):
    _run(tmp_path)
    db = sqlite3.connect(_db_path(tmp_path))
    settled = db.execute("SELECT task_id FROM task_memory_settlement").fetchall()
    traces = db.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
    assert len(settled) == traces == 1


def test_crash_between_scoring_and_extraction_is_recovered_on_resume(tmp_path):
    """Simulate: outcome written, extraction never ran, process died."""
    _run(tmp_path)
    path = _db_path(tmp_path)

    db = sqlite3.connect(path)
    experiment_ids = [r[0] for r in db.execute(
        "SELECT experience_id FROM experiences")]
    assert experiment_ids, "precondition: the first run learned something"
    # the crash leaves an outcome and a completed trace, but no settlement marker
    # and no experience
    db.execute("DELETE FROM task_memory_settlement")
    db.execute("DELETE FROM experience_events")
    db.execute("DELETE FROM experiences")
    db.commit()
    db.close()

    _run(tmp_path)          # resume into the same run directory

    db = sqlite3.connect(path)
    recovered = db.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
    traces = db.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
    settled = db.execute("SELECT COUNT(*) FROM task_memory_settlement").fetchone()[0]
    assert traces == 1, "the task must not be re-executed on resume"
    assert recovered >= 1, "the lesson must be re-derived, not silently dropped"
    assert settled == 1


def test_resume_does_not_relearn_a_settled_task(tmp_path):
    _run(tmp_path)
    path = _db_path(tmp_path)
    db = sqlite3.connect(path)
    before = db.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
    db.close()

    _run(tmp_path)

    db = sqlite3.connect(path)
    after = db.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
    assert after == before, "a settled task must not be extracted twice"


def test_paid_runs_cannot_silently_skip_the_shared_ledger(tmp_path, monkeypatch):
    """Regression: 90 paid calls once ran without the ledger because a flag was omitted."""
    from evomemory.runner import main

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-not-used")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-flash")
    code = main(["--modes", "A", "--llm", "api", "--per-family", "1",
                 "--families", "escalation", "--out", str(tmp_path)])
    assert code == 2, "a paid run without the ledger must be refused, not warned about"
