"""Paired statistics used to compare modes on the same tasks."""

import pytest

from evomemory.metrics import (
    compare_paired, mcnemar_exact, paired_bootstrap_ci,
)


def test_mcnemar_matches_the_binomial_tail():
    assert mcnemar_exact(0, 0) == 1.0          # no discordant pairs
    assert mcnemar_exact(5, 5) == 1.0          # perfectly balanced
    assert mcnemar_exact(10, 0) == pytest.approx(2 * 0.5 ** 10)
    assert mcnemar_exact(0, 10) == pytest.approx(2 * 0.5 ** 10)
    # one-sided information is preserved: all gains is as significant as all losses
    assert mcnemar_exact(9, 1) < 0.05
    assert mcnemar_exact(6, 2) > 0.05


def test_paired_bootstrap_ci_brackets_the_point_estimate():
    base = [False] * 20
    cand = [True] * 10 + [False] * 10
    lo, hi, delta = paired_bootstrap_ci(base, cand, n_boot=2000, seed=1)
    assert delta == pytest.approx(0.5)
    assert lo <= delta <= hi
    assert lo >= 0.0  # never negative when nothing got worse


def test_paired_bootstrap_rejects_unpaired_input():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([True], [True, False])


def test_compare_paired_counts_gains_and_losses():
    base = {"t1": False, "t2": False, "t3": True}
    cand = {"t1": True, "t2": True, "t3": True}
    out = compare_paired(base, cand)
    assert out["gained"] == 2 and out["lost"] == 0
    assert out["delta"] == pytest.approx(2 / 3, abs=1e-3)  # rounded to 4dp
    # two discordant pairs is p = 0.5: not evidence, and the report must say so
    assert out["significant_at_05"] is False


def test_compare_paired_is_significant_when_gains_dominate():
    base = {f"t{i}": False for i in range(10)}
    cand = {f"t{i}": True for i in range(8)}
    cand.update({f"t{i}": False for i in range(8, 10)})
    out = compare_paired(base, cand)
    assert out["gained"] == 8 and out["lost"] == 0
    assert out["significant_at_05"] is True
    assert out["mcnemar_p"] == pytest.approx(2 * 0.5 ** 8, abs=1e-5)  # rounded to 6dp


def test_compare_paired_only_uses_shared_tasks():
    base = {"t1": False, "t2": False}
    cand = {"t1": True, "t9": True}
    out = compare_paired(base, cand)
    assert out["n_pairs"] == 1


def test_compare_paired_empty():
    assert compare_paired({}, {})["n_pairs"] == 0
