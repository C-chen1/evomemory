"""A visual task environment that can be graded without a model in the loop.

The point of this environment is to make *seeing* necessary. Everything that can
leak the answer through text is therefore removed on purpose:

* the goal text never states a lamp colour or a readout value;
* ``list_zones`` returns zone **names only** — it deliberately omits lamp state
  and readout, which exist only in the pixels;
* the image carries no caption in its metadata, and is stored under a
  content-addressed name that says nothing about its contents;
* grading compares switch states against the ground truth computed at generation
  time, never against anything the model said.

What the task asks: switch on every zone whose lamp is red — except zones in
**lockdown**, which are red *and* have a readout >= 50. The lockdown rule is a
conjunction over two visual facts, mirroring the textual benchmark's hardest
case but with the operands available only through vision.

Honest limitation, stated here rather than discovered later: this environment has
**no live-model results**. The API call budget for the round was exhausted during
the textual ablation. The machinery below is unit-tested with a deterministic
stand-in perceiver; the vision path itself is exercised only by
``scripts/vision_experiment.py``, which the user must run.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

from .models import ExecutionTrace, Outcome, OutcomeStatus, Task, VerifiedBy

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _panel_font(size: int):
    """Load a scalable font on Linux and Windows without changing saved panels."""
    from PIL import ImageFont

    for font in (FONT, "DejaVuSans-Bold.ttf", "arialbd.ttf"):
        try:
            return ImageFont.truetype(font, size)
        except OSError:
            continue
    # Recent Pillow versions bundle a scalable default font.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()
LAMP_COLORS = {
    "green": (40, 170, 70),
    "amber": (230, 160, 30),
    "red": (200, 40, 40),
}
LOCKDOWN_READOUT = 50


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


@dataclass
class Zone:
    name: str
    lamp: str          # green | amber | red
    readout: int       # 0..99
    switch: str = "off"   # off | on

    @property
    def is_red(self) -> bool:
        return self.lamp == "red"

    @property
    def is_lockdown(self) -> bool:
        """The conjunction the agent has to learn: red AND high readout."""
        return self.lamp == "red" and self.readout >= LOCKDOWN_READOUT

    @property
    def is_target(self) -> bool:
        return self.lamp == "red" and not self.is_lockdown


@dataclass
class PanelTask:
    task: Task
    zones: List[Zone]
    image: bytes
    expected_on: List[str] = field(default_factory=list)
    lockdown: List[str] = field(default_factory=list)
    untouched: List[str] = field(default_factory=list)

    def facts(self) -> Dict[str, Dict[str, Any]]:
        """Per-zone visual facts, as the predicate engine sees them."""
        return {z.name: {"lamp": z.lamp, "readout": z.readout} for z in self.zones}


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def render_panel(zones: Sequence[Zone], title: str = "CONTROL PANEL") -> bytes:
    """Draw the panel. No text annotation of state beyond the lamp and readout."""
    from PIL import Image, ImageDraw

    width, height = 720, 140 + 150 * len(zones)
    img = Image.new("RGB", (width, height), (246, 246, 248))
    draw = ImageDraw.Draw(img)
    draw.rectangle([12, 12, width - 12, height - 12], outline=(40, 40, 60), width=3)
    draw.text((32, 26), title, font=_panel_font(30), fill=(25, 25, 45))

    label_font = _panel_font(34)
    read_font = _panel_font(40)
    for i, zone in enumerate(zones):
        top = 96 + i * 150
        draw.rectangle([32, top, width - 32, top + 120], outline=(150, 150, 170), width=2)
        draw.text((52, top + 36), zone.name, font=label_font, fill=(20, 20, 20))
        # lamp
        draw.ellipse([150, top + 34, 202, top + 86], fill=LAMP_COLORS[zone.lamp],
                     outline=(30, 30, 30), width=2)
        # readout box
        draw.rectangle([240, top + 24, 380, top + 96], outline=(40, 40, 60), width=2)
        draw.text((266, top + 34), f"{zone.readout:02d}", font=read_font,
                  fill=(10, 10, 10))
        # switch indicator: hollow = off, filled = on
        box = [440, top + 34, 520, top + 86]
        if zone.switch == "on":
            draw.rectangle(box, fill=(60, 120, 200), outline=(30, 30, 30), width=2)
        else:
            draw.rectangle(box, outline=(120, 120, 140), width=2)

    buf = io.BytesIO()
    img.save(buf, format="PNG")     # deliberately no pnginfo / caption
    return buf.getvalue()


# --------------------------------------------------------------------------
# task generation
# --------------------------------------------------------------------------


def _lcg(seed: int):
    state = seed & 0x7FFFFFFF

    def nxt(lo: int, hi: int) -> int:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return lo + state % (hi - lo + 1)

    return nxt


def build_panel_task(seed: int, task_id: str = "", n_zones: int = 3) -> PanelTask:
    """Deterministic panel with at least one target, one lockdown, one idle zone."""
    nxt = _lcg(seed)
    zones: List[Zone] = [Zone(name=chr(ord("A") + i), lamp="green", readout=0)
                         for i in range(n_zones)]

    # guaranteed composition so the task is neither trivial nor impossible
    zones[0].lamp, zones[0].readout = "red", nxt(10, LOCKDOWN_READOUT - 1)      # target
    zones[1].lamp, zones[1].readout = "red", nxt(LOCKDOWN_READOUT, 95)          # lockdown
    zones[2].lamp, zones[2].readout = nxt(0, 1) and "green" or "amber", nxt(5, 90)
    for i in range(3, n_zones):
        zones[i].lamp = ("green", "amber", "red")[nxt(0, 2)]
        zones[i].readout = nxt(1, 95)

    expected = [z.name for z in zones if z.is_target]
    lockdown = [z.name for z in zones if z.is_lockdown]
    untouched = [z.name for z in zones if not z.is_target]

    task = Task(
        goal=("The panel below shows three or more zones, each with a lamp and a "
              "readout. Switch ON every zone whose lamp is red, then write one log "
              "entry. Read the panel image: the zone states are not available "
              "through any tool."),
        task_id=task_id or f"panel_{seed}",
        context={"seed": seed, "panel": "control", "zones": [z.name for z in zones]},
        tags=["visual", "panel"],
        family="panel",
        max_steps=n_zones * 3 + 8,
    )
    return PanelTask(task=task, zones=zones, image=render_panel(zones),
                     expected_on=expected, lockdown=lockdown, untouched=untouched)


def build_panel_task_v2(seed: int, task_id: str, n_zones: int = 3) -> PanelTask:
    """Versioned panel generator that does not assign a class to a fixed label.

    This is a new dataset, not a byte-identical replacement for historical data.
    Seeds are kept out of the public task context; split membership is managed by
    the experiment, using disjoint seeds and identifiers.
    """
    import random

    if not task_id:
        raise ValueError("a non-empty, split-specific task ID is required")
    bt = build_panel_task(seed, task_id=task_id, n_zones=n_zones)
    states = [(z.lamp, z.readout) for z in bt.zones]
    random.Random(seed ^ 0xE701).shuffle(states)
    for z, (lamp, readout) in zip(bt.zones, states):
        z.lamp, z.readout = lamp, readout
    bt.task.context = {"panel": "control", "zones": [z.name for z in bt.zones],
                       "dataset_version": "panel-v2"}
    bt.image = render_panel(bt.zones)
    bt.expected_on = [z.name for z in bt.zones if z.is_target]
    bt.lockdown = [z.name for z in bt.zones if z.is_lockdown]
    bt.untouched = [z.name for z in bt.zones if not z.is_target]
    return bt


def build_panel_task_v3(seed: int, task_id: str, n_zones: int = 3) -> PanelTask:
    """Versioned task wording consistent with unspecified operating restrictions."""
    bt = build_panel_task_v2(seed, task_id, n_zones)
    bt.task.goal = ("The panel below shows three or more zones, each with a lamp and a "
                    "readout. Switch ON red-lamp zones where operating restrictions "
                    "permit, then write one log entry. Read the panel image: the zone "
                    "states are not available through any tool.")
    bt.task.context["dataset_version"] = "panel-v3"
    return bt


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------


class PanelEnv:
    """Tools that expose zone *names* and let switches be flipped — nothing else.

    Note what is missing: there is no way to ask for a zone's lamp or readout.
    If such a tool existed, the image would be decorative and the experiment would
    measure nothing.
    """

    TOOLS = ("list_zones", "set_switch", "log_entry")

    def __init__(self, bt: PanelTask):
        self.bt = bt
        self.switches: Dict[str, str] = {z.name: "off" for z in bt.zones}
        self.log: List[str] = []
        self.calls = 0

    def spec(self) -> List[Dict[str, Any]]:
        return [
            {"name": "list_zones", "args": {},
             "returns": "zone names only — lamp state and readout are shown in the image"},
            {"name": "set_switch", "args": {"zone": "str", "state": "on|off"},
             "returns": "updated switch state"},
            {"name": "log_entry", "args": {"text": "str"}, "returns": "ok"},
        ]

    def call(self, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        self.calls += 1
        handler = getattr(self, f"_tool_{tool}", None)
        if handler is None or tool not in self.TOOLS:
            return {"ok": False, "error": f"unknown tool {tool!r}",
                    "available": list(self.TOOLS)}
        try:
            return {"ok": True, "result": handler(args or {})}
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the agent
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _tool_list_zones(self, args: Dict[str, Any]) -> List[str]:
        return [z.name for z in self.bt.zones]

    def _tool_set_switch(self, args: Dict[str, Any]) -> Dict[str, Any]:
        zone = args.get("zone")
        if zone not in self.switches:
            raise KeyError(f"no zone {zone!r}")
        # Accept the obvious aliases. A model that writes {"zone": "A", "on": "on"}
        # is not wrong about the world, only about the key name; failing it teaches
        # nothing and burns the step budget on retries.
        raw = args.get("state", args.get("on", args.get("value")))
        if isinstance(raw, bool):
            state = "on" if raw else "off"
        else:
            state = str(raw).strip().lower()
            if state in ("true", "1", "yes"):
                state = "on"
            elif state in ("false", "0", "no"):
                state = "off"
        if state not in ("on", "off"):
            raise ValueError("state must be 'on' or 'off'")
        self.switches[zone] = state
        return {"zone": zone, "state": state, "switches": dict(self.switches)}

    def _tool_log_entry(self, args: Dict[str, Any]) -> str:
        text = str(args.get("text", "")).strip()
        if not text:
            raise ValueError("log text is empty")
        self.log.append(text)
        return "ok"


def verify(bt: PanelTask, env: PanelEnv) -> Outcome:
    """Exact switch comparison against the state computed at generation time."""
    checks: List[Dict[str, Any]] = []

    missing = [z for z in bt.expected_on if env.switches[z] != "on"]
    checks.append({"name": "targets_switched_on", "passed": not missing,
                   "detail": f"{len(bt.expected_on) - len(missing)}/{len(bt.expected_on)} on"
                             + (f"; missing {missing}" if missing else "")})

    touched = [z for z in bt.lockdown if env.switches[z] == "on"]
    checks.append({"name": "lockdown_respected", "passed": not touched,
                   "detail": "no lockdown zone switched on" if not touched
                             else f"lockdown zones switched on: {touched}"})

    idle = [z for z in bt.untouched if z not in bt.lockdown and env.switches[z] == "on"]
    checks.append({"name": "no_collateral_switching", "passed": not idle,
                   "detail": "no unrelated zone switched" if not idle
                             else f"unexpectedly switched: {idle}"})

    checks.append({"name": "log_written", "passed": bool(env.log),
                   "detail": f"{len(env.log)} log entries"})

    passed = all(c["passed"] for c in checks)
    return Outcome(
        status=OutcomeStatus.SUCCESS if passed else OutcomeStatus.FAILURE,
        verified_by=VerifiedBy.PROGRAM,
        score=sum(1 for c in checks if c["passed"]) / len(checks),
        reason="all checks passed" if passed else
               "; ".join(c["detail"] for c in checks if not c["passed"]),
        checks=checks,
    )


# --------------------------------------------------------------------------
# learning the rule from a failure (ground-truth side of the task)
# --------------------------------------------------------------------------


def failure_counterexamples(bt: PanelTask, env: PanelEnv) -> List[List[tuple[str, Any]]]:
    """Counterexamples in the same grouped shape the textual benchmark produces.

    One record violating both attributes is what tells the extractor that the rule
    is a conjunction rather than a disjunction.
    """
    out: List[List[tuple[str, Any]]] = []
    for z in bt.zones:
        if z.name in bt.lockdown and env.switches[z.name] == "on":
            out.append([("lamp", z.lamp), ("readout_high", True)])
    return out


def structured_training_feedback(bt: PanelTask, env: PanelEnv, outcome: Outcome,
                                 *, run_id: str, image_evidence_id: str,
                                 split: str, evidence_store, trace: ExecutionTrace,
                                 agent_observation: Any = None,
                                 model_interpretation: Any = None) -> str:
    """Persist post-task supervised feedback and return its evidence ID.

    The caller supplies the completed, program-verified outcome. Holdout feedback
    must never enter the learning path. Agent statements remain separate from
    scorer ground truth; an omitted statement is recorded as unknown.
    """
    if split != "train":
        raise ValueError("feedback is restricted to training tasks")
    if not run_id or not image_evidence_id or not evidence_store.exists(image_evidence_id):
        raise ValueError("run and image evidence provenance required")
    if (trace.status != "completed" or trace.task_id != bt.task.task_id or
            trace.run_id != run_id):
        raise ValueError("a completed trace matching the task and run is required")
    recorded = evidence_store.db.query_one(
        "SELECT t.task_id, t.run_id, t.status, o.status AS outcome_status, "
        "o.verified_by, o.checks_json FROM traces t JOIN outcomes o ON o.trace_id = t.trace_id "
        "WHERE t.trace_id = ?", (trace.trace_id,))
    from .db import loads
    if (recorded is None or recorded["status"] != "completed" or
            recorded["task_id"] != bt.task.task_id or recorded["run_id"] != run_id or
            recorded["outcome_status"] != outcome.status.value or
            recorded["verified_by"] != VerifiedBy.PROGRAM.value or
            loads(recorded["checks_json"], []) != outcome.checks):
        raise ValueError("completed program-verified trace is missing or inconsistent")
    import hashlib
    image = evidence_store.get(image_evidence_id)
    if (image.kind.value != "image" or image.sha256 != hashlib.sha256(bt.image).hexdigest() or
            evidence_store.read_bytes(image_evidence_id) != bt.image):
        raise ValueError("image evidence does not match the task")
    if env.bt is not bt or outcome.verified_by is not VerifiedBy.PROGRAM:
        raise ValueError("feedback requires this task's program-verified result")
    actual = verify(bt, env)
    if outcome.to_dict() != actual.to_dict():
        raise ValueError("outcome does not match the final environment")
    rows = []
    for z in bt.zones:
        observed_action = "activate" if env.switches[z.name] == "on" else "leave_off"
        expected_action = "activate" if z.is_target else "leave_off"
        rows.append({"zone": z.name, "ground_truth": {"lamp": z.lamp, "readout": z.readout},
                     "action": observed_action, "expected_action": expected_action,
                     "outcome": "correct" if observed_action == expected_action else "failed"})
    payload = {"task_id": bt.task.task_id, "run_id": run_id, "trace_id": trace.trace_id,
               "image_evidence_id": image_evidence_id, "split": split,
               "agent_observation": agent_observation,
               "model_interpretation": model_interpretation,
               "scorer_outcome": outcome.to_dict(), "zones": rows}
    return evidence_store.put_json(payload, metadata={"kind": "structured_training_feedback"},
                                   derived_from=image_evidence_id).evidence_id


def visual_condition() -> Dict[str, Any]:
    """The rule an agent should end up with, as an executable predicate."""
    return {"all_of": [{"field": "lamp", "op": "eq", "value": "red"},
                       {"field": "readout", "op": "gte", "value": LOCKDOWN_READOUT}]}
