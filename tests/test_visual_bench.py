"""Visual environment: determinism, no text leakage, and real pixels."""

import io

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from evomemory.visual_bench import (
    LAMP_COLORS, LOCKDOWN_READOUT, PanelEnv, build_panel_task, render_panel,
    verify, visual_condition,
)
from evomemory.predicate import Tri, evaluate, validate


def test_generation_is_deterministic():
    a = build_panel_task(seed=11)
    b = build_panel_task(seed=11)
    assert [(z.name, z.lamp, z.readout) for z in a.zones] == \
           [(z.name, z.lamp, z.readout) for z in b.zones]
    assert a.image == b.image


def test_every_task_has_a_target_a_lockdown_and_an_idle_zone():
    for seed in range(20):
        bt = build_panel_task(seed=seed)
        assert bt.expected_on, f"seed {seed}: no target"
        assert bt.lockdown, f"seed {seed}: no lockdown zone"
        assert len(bt.untouched) >= 1


def test_lockdown_is_a_strict_subset_of_red():
    bt = build_panel_task(seed=5)
    reds = {z.name for z in bt.zones if z.is_red}
    lockdown = set(bt.lockdown)
    assert lockdown <= reds
    assert set(bt.expected_on) == reds - lockdown


# -- leakage guards --------------------------------------------------------


def test_goal_text_leaks_neither_values_nor_the_hidden_rule():
    """The goal may state the observable criterion ("red"), never the exception.

    Stating "red means fault" is the *task*. What must not leak is the lockdown
    conjunction: the readout threshold, the word lockdown, or which zones are in
    it. Without those, the only way to decide is to look at the image.
    """
    bt = build_panel_task(seed=7)
    goal = bt.task.goal.lower()
    assert "lockdown" not in goal
    assert str(LOCKDOWN_READOUT) not in goal
    for zone in bt.zones:
        assert str(zone.readout) not in goal
        assert f"{zone.readout:02d}" not in goal
    # zone *names* are public (list_zones reveals them), so the real requirement
    # is that no zone is identified as being in lockdown
    assert not any(phrase in goal for phrase in
                   ("zone a", "zone b", "zone c", "except", "skip"))


def test_list_zones_hides_state():
    bt = build_panel_task(seed=3)
    env = PanelEnv(bt)
    out = env.call("list_zones", {})
    assert out["ok"]
    assert out["result"] == [z.name for z in bt.zones]
    # nothing about lamps or readouts may appear in any tool output
    blob = str(out)
    for colour in LAMP_COLORS:
        assert colour not in blob
    for zone in bt.zones:
        assert str(zone.readout) not in blob


def test_image_carries_no_caption_metadata():
    bt = build_panel_task(seed=9)
    with Image.open(io.BytesIO(bt.image)) as img:
        assert not (img.info or {}).get("Description")
        assert not (img.info or {}).get("caption")


def test_no_tool_can_reveal_state():
    bt = build_panel_task(seed=4)
    env = PanelEnv(bt)
    names = {t["name"] for t in env.spec()}
    assert names == {"list_zones", "set_switch", "log_entry"}
    assert "get_record" not in names and "read_panel" not in names


# -- the pixels really do carry the state ----------------------------------


def test_lamp_colour_is_present_in_the_image():
    bt = build_panel_task(seed=2)
    with Image.open(io.BytesIO(bt.image)) as img:
        rgb = img.convert("RGB")
        for i, zone in enumerate(bt.zones):
            # centre of the lamp circle for row i
            x, y = 176, 96 + i * 150 + 60
            pixel = rgb.getpixel((x, y))
            expected = LAMP_COLORS[zone.lamp]
            distance = sum(abs(a - b) for a, b in zip(pixel, expected))
            assert distance < 120, f"zone {zone.name}: got {pixel}, expected ~{expected}"


def test_readout_digits_are_rendered():
    """A cheap structural check: the readout box is not blank."""
    bt = build_panel_task(seed=6)
    with Image.open(io.BytesIO(bt.image)) as img:
        rgb = img.convert("RGB")
        for i in range(len(bt.zones)):
            box = (250, 96 + i * 150 + 34, 370, 96 + i * 150 + 90)
            crop = rgb.crop(box)
            raw = crop.tobytes()
            dark = sum(1 for i in range(0, len(raw), 3) if sum(raw[i:i + 3]) < 200)
            assert dark > 40, f"zone {i}: readout looks blank"


# -- grading ---------------------------------------------------------------


def test_do_nothing_fails():
    bt = build_panel_task(seed=1)
    assert verify(bt, PanelEnv(bt)).status.value == "failure"


def test_switching_only_targets_succeeds():
    bt = build_panel_task(seed=8)
    env = PanelEnv(bt)
    for zone in bt.expected_on:
        env.call("set_switch", {"zone": zone, "state": "on"})
    env.call("log_entry", {"text": "done"})
    outcome = verify(bt, env)
    assert outcome.status.value == "success", outcome.reason


def test_switching_a_lockdown_zone_fails():
    bt = build_panel_task(seed=8)
    env = PanelEnv(bt)
    for zone in bt.expected_on + bt.lockdown:
        env.call("set_switch", {"zone": zone, "state": "on"})
    env.call("log_entry", {"text": "done"})
    outcome = verify(bt, env)
    assert outcome.status.value == "failure"
    detail = next(c["detail"] for c in outcome.checks if c["name"] == "lockdown_respected")
    assert "lockdown" in detail


def test_unknown_zone_is_refused_not_raised():
    bt = build_panel_task(seed=1)
    env = PanelEnv(bt)
    assert env.call("set_switch", {"zone": "Z", "state": "on"})["ok"] is False
    assert env.call("set_switch", {"zone": "A", "state": "maybe"})["ok"] is False


# -- the learned rule is the same predicate the text benchmark would use ----


def test_visual_condition_matches_ground_truth():
    cond = visual_condition()
    validate(cond, {"lamp", "readout"})
    assert evaluate(cond, {"lamp": "red", "readout": 60}) is Tri.TRUE
    assert evaluate(cond, {"lamp": "red", "readout": 49}) is Tri.FALSE
    assert evaluate(cond, {"lamp": "amber", "readout": 60}) is Tri.FALSE
    assert evaluate(cond, {"lamp": "red"}) is Tri.UNKNOWN   # missing readout
