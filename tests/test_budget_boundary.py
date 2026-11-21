"""Every remote call must go through BudgetedLLM with the round ledger.

A diagnostic script used to call urllib directly and another built a paid client
without a shared budget, so money could be spent that the round never saw. Those
paths are gone; this test keeps them gone without making any network call.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(p for p in (ROOT / "scripts").glob("*.py"))


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_no_script_issues_raw_http_requests():
    offenders = [p.name for p in SCRIPTS
                 if re.search(r"urlopen\(|requests\.(get|post)\(", _source(p))]
    assert not offenders, f"raw HTTP in {offenders}: use BudgetedLLM instead"


def test_every_paid_client_is_wrapped_in_a_budgeted_client_with_the_ledger():
    for path in SCRIPTS:
        src = _source(path)
        if "DeepSeekClient(" not in src:
            continue
        # A paid client may be built inline or bound to a variable, but it must
        # always end up inside a BudgetedLLM that is attached to the round ledger.
        assert "BudgetedLLM(" in src, \
            f"{path.name}: a paid client must be wrapped in BudgetedLLM"
        assert "shared_budget=" in src, \
            f"{path.name}: paid calls must reference the shared ledger"
        assert "artifacts" in src and "round_budget.db" in src, \
            f"{path.name}: paid calls must point at the round ledger file"


def test_runner_has_no_unbudgeted_path():
    src = _source(ROOT / "src" / "evomemory" / "runner.py")
    assert "allow-unbudgeted" not in src and "allow_unbudgeted" not in src
    assert "without --shared-stage3-budget" in src


def test_shared_budget_module_is_the_single_ledger():
    src = _source(ROOT / "src" / "evomemory" / "shared_budget.py")
    assert "budget_calls" in src and "reserve" in src and "settle" in src
