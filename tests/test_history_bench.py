"""Offline checks for the history-dependent visual task.

The point of this environment is that the answer is *not* in the current image.
These tests are the evidence for that claim, and for the claim that a memory with
traced history can act on it.
"""

import hashlib

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from evomemory.history_bench import (
    LOCK_AFTER, REGIONS, act_image_only, act_with_history, build_sequence,
    render_history_panel, to_env, verify,
)


@pytest.fixture()
def seq():
    return build_sequence(seed=1234, n_tasks=4)


# -- 1. the current image really does carry the current state ---------------


def test_lamp_colour_is_in_the_pixels(seq):
    panel = seq[0]
    with Image.open(__import__("io").BytesIO(panel.image)) as img:
        rgb = img.convert("RGB")
        for i, zone in enumerate(panel.zones):
            x, y = 176, 96 + i * 150 + 60
            pixel = rgb.getpixel((x, y))
            expected = (40, 170, 70) if zone.lamp == "green" else (200, 40, 40)
            assert sum(abs(a - b) for a, b in zip(pixel, expected)) < 120


def test_generation_is_deterministic():
    a = build_sequence(seed=77, n_tasks=3)
    b = build_sequence(seed=77, n_tasks=3)
    assert [p.image for p in a] == [p.image for p in b]
    assert [p.history for p in a] == [p.history for p in b]


# -- 2. the history is NOT reachable from the task as given ----------------


def test_goal_text_says_nothing_about_history(seq):
    for panel in seq:
        goal = panel.task.goal.lower()
        for region in REGIONS:
            assert region not in goal
        for n in ("0", "1", "2", "3"):
            assert f"{n} times" not in goal
        assert "count" not in goal and "previous" not in goal


def test_no_tool_can_reveal_history(seq):
    env = to_env(seq[0])
    names = {t["name"] for t in env.spec()}
    assert names == {"list_zones", "set_switch", "log_entry", "skip_zone"}
    blob = str(env.call("list_zones", {}))
    for region in REGIONS:
        value = seq[0].history[region]
        if value:
            assert str(value) not in blob


def test_the_image_does_not_encode_a_counter(seq):
    """Two panels with the same current lamps but different history must look alike
    in terms of what they show: only the lamp colour is rendered."""
    panel = seq[2]
    same = type(panel)(task=panel.task, zones=list(panel.zones),
                       image=render_history_panel(panel.zones),
                       history={r: 0 for r in REGIONS}, locked=[],
                       expected_on=[])
    assert same.image == panel.image, "history must not change the pixels"
    assert same.history != panel.history


# -- 3/4. the baseline cannot solve it; history can ------------------------


def test_image_only_baseline_fails_exactly_when_a_region_is_locked(seq):
    outcomes = []
    for panel in seq:
        switches, log = act_image_only(panel)
        outcomes.append(verify(panel, switches, log).status.value)
    assert outcomes[:2] == ["success", "success"], "no locks yet"
    assert outcomes[2:] == ["failure", "failure"], "locked regions over-acted"


def test_history_aware_policy_solves_every_panel(seq):
    for panel in seq:
        switches, log = act_with_history(panel)
        outcome = verify(panel, switches, log)
        assert outcome.status.value == "success", (panel.task.task_id, outcome.reason)


def test_lock_threshold_is_reached_during_the_sequence(seq):
    assert seq[0].locked == [] and seq[1].locked == []
    assert seq[2].locked, "the sequence must actually produce a locked region"


# -- 5. wrong or stale history causes negative transfer --------------------


def test_stale_history_causes_over_action(seq):
    panel = seq[2]
    panel.history = {r: 0 for r in REGIONS}          # memory claims no prior sightings
    switches, log = act_with_history(panel)
    outcome = verify(panel, switches, log)
    assert outcome.status.value == "failure"
    assert "locked" in outcome.reason


def test_inflated_history_causes_over_exclusion(seq):
    panel = seq[0]                                    # nothing is actually locked yet
    panel.history = {r: 5 for r in REGIONS}
    switches, log = act_with_history(panel)
    outcome = verify(panel, switches, log)
    assert outcome.status.value == "failure"
    assert "missing" in outcome.reason


# -- 6. provenance ---------------------------------------------------------


def test_images_are_distinct_and_content_addressable(seq):
    hashes = [hashlib.sha256(p.image).hexdigest() for p in seq]
    assert len(set(hashes)) == len(hashes), "each panel is its own evidence"
    assert all(len(h) == 64 for h in hashes)


def test_history_is_a_prefix_of_the_observed_red_counts(seq):
    """The supplied history must match what the sequence actually made observable."""
    reds = {r: 0 for r in REGIONS}
    for panel in seq:
        assert panel.history == reds, "history must equal prior observed reds"
        for z in panel.zones:
            if z.lamp == "red":
                reds[z.name] += 1
