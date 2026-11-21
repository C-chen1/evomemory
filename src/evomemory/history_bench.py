"""A visual task whose answer is NOT derivable from the current image.

Why another environment: the earlier panel task failed as a benchmark in both
directions. With "open every red zone" the wording contradicted the grader, so a
memory that said "don't" looked helpful for the wrong reason. With "where
operating restrictions permit" the restriction (readout >= 50) became inferable
from the text plus the pixels, so an agent with no memory solved it 2/2.

Here the restriction depends only on **how many times a region has been red in
earlier tasks**. That count is not in the current image, not in the goal text, and
not available from any tool. It exists only in the run history, so:

* an agent that looks only at the current panel must over-act on locked regions;
* an agent that can consult traced history can avoid exactly those regions.

Everything is still graded programmatically, and the history is deliberately kept
out of the task record: it is supplied to the policy from the sequence, which is
what an experience-memory layer would have to reconstruct from stored evidence.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import Outcome, OutcomeStatus, Task, VerifiedBy
from .visual_bench import FONT, LAMP_COLORS, PanelEnv, PanelTask, Zone, render_panel

#: A region that has been red this many times before is locked.
LOCK_AFTER = 2
REGIONS = ("north", "south", "east")


@dataclass
class HistoryPanel:
    task: Task
    zones: List[Zone]                 # zone.name is the region name
    image: bytes
    history: Dict[str, int]           # red count *before* this panel
    locked: List[str] = field(default_factory=list)
    expected_on: List[str] = field(default_factory=list)

    def facts(self) -> Dict[str, Dict[str, Any]]:
        return {z.name: {"lamp": z.lamp} for z in self.zones}


def _lcg(seed: int):
    state = seed & 0x7FFFFFFF

    def nxt(lo: int, hi: int) -> int:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return lo + state % (hi - lo + 1)

    return nxt


def render_history_panel(zones: Sequence[Zone]) -> bytes:
    """Same renderer as the other panel family; region names are the zone labels."""
    return render_panel(zones, title="REGION STATUS")


def build_sequence(seed: int, n_tasks: int = 4) -> List[HistoryPanel]:
    """A deterministic sequence where region redness accumulates across tasks."""
    nxt = _lcg(seed)
    reds: Dict[str, int] = {r: 0 for r in REGIONS}
    panels: List[HistoryPanel] = []
    for i in range(n_tasks):
        zones: List[Zone] = []
        for region in REGIONS:
            # bias a couple of regions towards being red repeatedly
            hot = region in ("north", "east")
            is_red = (nxt(0, 2) == 0) if not hot else (nxt(0, 3) != 0)
            zones.append(Zone(name=region, lamp="red" if is_red else "green",
                              readout=nxt(10, 90)))
        history = dict(reds)
        locked = [z.name for z in zones if z.lamp == "red" and history[z.name] >= LOCK_AFTER]
        expected = [z.name for z in zones if z.lamp == "red" and z.name not in locked]
        for z in zones:
            if z.lamp == "red":
                reds[z.name] += 1
        task = Task(
            # No mention of counts, history or earlier inspections: saying so would
            # hint that history matters, which is the thing being tested.
            goal=("The panel shows the current lamp colour of each region. Switch ON "
                  "every red region that is still permitted, then write one log entry."),
            task_id=f"history-{i:02d}",
            context={"seed": seed, "index": i},
            tags=["visual", "history"], family="history",
            max_steps=len(zones) * 3 + 8,
        )
        panels.append(HistoryPanel(task=task, zones=zones,
                                   image=render_history_panel(zones),
                                   history=history, locked=locked,
                                   expected_on=expected))
    return panels


def verify(panel: HistoryPanel, switches: Dict[str, str], log: Sequence[str]) -> Outcome:
    checks: List[Dict[str, Any]] = []
    missing = [r for r in panel.expected_on if switches.get(r) != "on"]
    checks.append({"name": "targets_switched_on", "passed": not missing,
                   "detail": f"{len(panel.expected_on) - len(missing)}"
                             f"/{len(panel.expected_on)} on"
                             + (f"; missing {missing}" if missing else "")})
    touched = [r for r in panel.locked if switches.get(r) == "on"]
    checks.append({"name": "locked_respected", "passed": not touched,
                   "detail": "no locked region switched on" if not touched
                             else f"locked regions switched on: {touched}"})
    idle = [z.name for z in panel.zones
            if z.lamp != "red" and switches.get(z.name) == "on"]
    checks.append({"name": "no_collateral_switching", "passed": not idle,
                   "detail": "no green region switched" if not idle
                             else f"unexpectedly switched: {idle}"})
    checks.append({"name": "log_written", "passed": bool(log),
                   "detail": f"{len(log)} log entries"})
    passed = all(c["passed"] for c in checks)
    return Outcome(status=OutcomeStatus.SUCCESS if passed else OutcomeStatus.FAILURE,
                   verified_by=VerifiedBy.PROGRAM,
                   score=sum(1 for c in checks if c["passed"]) / len(checks),
                   reason="all checks passed" if passed else
                          "; ".join(c["detail"] for c in checks if not c["passed"]),
                   checks=checks)


# --------------------------------------------------------------------------
# two reference policies: what the task is supposed to discriminate
# --------------------------------------------------------------------------


def act_image_only(panel: HistoryPanel) -> Tuple[Dict[str, str], List[str]]:
    """The baseline: red means on. Cannot see the history, so it over-acts."""
    switches = {z.name: ("on" if z.lamp == "red" else "off") for z in panel.zones}
    return switches, ["switched on every red region"]


def act_with_history(panel: HistoryPanel) -> Tuple[Dict[str, str], List[str]]:
    """Uses only the traced history, never the grader's lock list."""
    switches: Dict[str, str] = {}
    for z in panel.zones:
        if z.lamp != "red":
            switches[z.name] = "off"
            continue
        switches[z.name] = "off" if panel.history[z.name] >= LOCK_AFTER else "on"
    return switches, ["applied the red-count rule from earlier inspections"]


def to_env(panel: HistoryPanel) -> PanelEnv:
    """Wrap a history panel in the standard panel environment (tools unchanged)."""
    bt = PanelTask(task=panel.task, zones=panel.zones, image=panel.image,
                   expected_on=list(panel.expected_on), lockdown=list(panel.locked),
                   untouched=[z.name for z in panel.zones if z.name not in panel.expected_on])
    return PanelEnv(bt)
