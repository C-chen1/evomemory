"""A deterministic tool-use benchmark with programmatic verification.

Why a custom micro-benchmark instead of ALFWorld / tau2 / WebShop:

* those need GPUs, Docker, or multi-gigabyte downloads, none of which exist here;
* their scoring depends on an LLM judge or a simulator, which makes "success"
  unfalsifiable at this scale;
* what EvoMemory needs to measure is narrow — does a lesson learned in task *n*
  change behaviour in task *n+1* of the same family, and does it hurt when the
  family differs.

So: a tiny "ops console" over a record table. Every check is an exact comparison
of environment state. No judge, no sampling noise.

Each task family has a **hidden rule** that is not stated in the goal. The agent
only learns it by failing once and receiving the failing check names. That is the
signal the experience lifecycle is supposed to capture and reuse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence

from .models import Outcome, OutcomeStatus, Task, VerifiedBy


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------


@dataclass
class Record:
    rid: str
    region: str
    tier: str          # gold | silver | bronze
    priority: int      # 1..5
    status: str        # open | escalated | closed
    note: str = ""


class OpsEnv:
    """A record table with four tools. State is fully inspectable."""

    TOOLS = ("list_records", "get_record", "set_field", "log_entry")

    def __init__(self, records: Sequence[Record]):
        self._initial = [Record(**asdict(r)) for r in records]
        self.records: Dict[str, Record] = {}
        self.log: List[str] = []
        self.call_count = 0
        self.reset()

    def reset(self) -> None:
        self.records = {r.rid: Record(**asdict(r)) for r in self._initial}
        self.log = []
        self.call_count = 0

    # -- tool surface ----------------------------------------------------
    def spec(self) -> List[Dict[str, Any]]:
        return [
            {"name": "list_records", "args": {"region": "str?", "tier": "str?",
                                              "min_priority": "int?", "status": "str?"},
             "returns": "list of record dicts"},
            {"name": "get_record", "args": {"rid": "str"}, "returns": "record dict"},
            {"name": "set_field", "args": {"rid": "str", "field": "str", "value": "any"},
             "returns": "updated record dict"},
            {"name": "log_entry", "args": {"text": "str"}, "returns": "ok"},
        ]

    def call(self, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        self.call_count += 1
        handler: Optional[Callable[[Dict[str, Any]], Any]] = getattr(
            self, f"_tool_{tool}", None)
        if handler is None or tool not in self.TOOLS:
            return {"ok": False, "error": f"unknown tool {tool!r}",
                    "available": list(self.TOOLS)}
        try:
            result = handler(args or {})
            return {"ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001 - surfaced to the agent verbatim
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _tool_list_records(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        out = []
        for r in self.records.values():
            if "region" in args and args["region"] not in (None, "", "any"):
                if r.region != args["region"]:
                    continue
            if "tier" in args and args["tier"] not in (None, "", "any"):
                if r.tier != args["tier"]:
                    continue
            if "status" in args and args["status"] not in (None, "", "any"):
                if r.status != args["status"]:
                    continue
            if "min_priority" in args and args["min_priority"] is not None:
                if r.priority < int(args["min_priority"]):
                    continue
            out.append(asdict(r))
        out.sort(key=lambda d: d["rid"])
        return out

    def _tool_get_record(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return asdict(self._require(args["rid"]))

    def _tool_set_field(self, args: Dict[str, Any]) -> Dict[str, Any]:
        r = self._require(args["rid"])
        fld = str(args["field"])
        if fld not in ("region", "tier", "priority", "status", "note"):
            raise ValueError(f"field {fld!r} is not writable")
        value = args["value"]
        if fld == "priority":
            value = int(value)
        setattr(r, fld, value)
        return asdict(r)

    def _tool_log_entry(self, args: Dict[str, Any]) -> str:
        text = str(args.get("text", "")).strip()
        if not text:
            raise ValueError("log text is empty")
        self.log.append(text)
        return "ok"

    def _require(self, rid: Any) -> Record:
        if not isinstance(rid, str) or rid not in self.records:
            raise KeyError(f"no record {rid!r}")
        return self.records[rid]

    # -- inspection -------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "records": {rid: asdict(r) for rid, r in sorted(self.records.items())},
            "log": list(self.log),
        }


# --------------------------------------------------------------------------
# hidden rules & tasks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HiddenRule:
    """The part of the task the goal text does not state."""

    name: str
    field: str            # record field the rule looks at
    blocked_values: tuple[str, ...]   # values that must NOT be acted on
    condition_field: str = ""         # optional: rule only applies when ...
    condition_value: str = ""

    def blocks(self, rec: Record) -> bool:
        if self.condition_field and getattr(rec, self.condition_field) != self.condition_value:
            return False
        return getattr(rec, self.field) in self.blocked_values

    def describe(self) -> str:
        cond = (f" when {self.condition_field}={self.condition_value}"
                if self.condition_field else "")
        return f"do not act on records with {self.field} in {list(self.blocked_values)}{cond}"


@dataclass
class Family:
    name: str
    action: str            # escalate | close
    goal_template: str
    rules: List[HiddenRule]
    tags: List[str] = field(default_factory=list)

    def excluded(self, rec: Record) -> bool:
        return any(r.blocks(rec) for r in self.rules)


FAMILIES: Dict[str, Family] = {
    "escalation": Family(
        name="escalation",
        action="escalate",
        goal_template=(
            "Escalate every open record in region {region} with priority >= {priority}. "
            "Use set_field with status='escalated', then write one log entry summarising "
            "what you did."),
        rules=[
            HiddenRule("gold_tier", "tier", ("gold",)),
            HiddenRule("hold_note", "note", ("hold",)),
        ],
        tags=["ops", "records"],
    ),
    "closure": Family(
        name="closure",
        action="close",
        goal_template=(
            "Close every escalated record in region {region} with priority <= {priority}. "
            "Use set_field with status='closed', then write one log entry summarising "
            "what you did."),
        rules=[
            HiddenRule("vip_tier", "tier", ("gold",)),
            HiddenRule("audit_note", "note", ("audit",)),
        ],
        tags=["ops", "records"],
    ),
    "conditional": Family(
        name="conditional",
        action="escalate",
        goal_template=(
            "Escalate every open record in region {region} with priority >= {priority}. "
            "Use set_field with status='escalated', then write one log entry summarising "
            "what you did."),
        # A conjunction, not a single-attribute veto. The two decoys make
        # over-generalisation observable: a lesson that blocks "all gold" or
        # "all audit" will skip records it should have acted on.
        rules=[
            HiddenRule("gold_audit", "tier", ("gold",),
                       condition_field="note", condition_value="audit"),
        ],
        tags=["ops", "records"],
    ),
}


TYPE_VALUES = {
    "gold": "gold",
    "silver": "silver",
    "bronze": "bronze",
}


@dataclass
class BenchTask:
    task: Task
    family: str
    region: str
    priority: int
    action: str
    rules: List[HiddenRule]
    expected_ids: List[str] = field(default_factory=list)
    excluded_ids: List[str] = field(default_factory=list)
    untouched_ids: List[str] = field(default_factory=list)
    initial_status: Dict[str, str] = field(default_factory=dict)

    @property
    def target_status(self) -> str:
        return "escalated" if self.action == "escalate" else "closed"


def build_table(seed: int, action: str, region: str, priority: int,
                family: str = "") -> List[Record]:
    """Deterministic record table with a *controlled* composition.

    An earlier version generated tables at random and produced tasks with zero
    targets, which are trivially passable and therefore worthless as evidence.
    Here every table contains, by construction:

    * 2-3 records that must be acted on,
    * 1-2 records that look like targets but are blocked by a hidden rule,
    * 3-4 near-misses that must not be touched (wrong region, wrong threshold,
      already in the desired state).
    """
    n_target = 2 + (seed % 2)
    recs: List[Record] = []
    idx = 0

    def add(region_: str, tier: str, prio: int, status: str, note: str = "") -> None:
        nonlocal idx
        recs.append(Record(f"r{idx:02d}", region_, tier, prio, status, note))
        idx += 1

    if family == "conditional":
        # two plain targets, a blocked conjunction, and two decoys that each
        # satisfy only *one* half of the conjunction
        for i in range(n_target):
            add(region, "silver" if i % 2 == 0 else "bronze", priority + (i % 2), "open")
        add(region, "gold", priority, "open", "audit")       # blocked: gold AND audit
        add(region, "gold", priority, "open")                # decoy: gold only -> act
        add(region, "silver", priority, "open", "audit")     # decoy: audit only -> act
        add("zz", "gold", 5, "open", "audit")                # wrong region
        add(region, "silver", max(1, priority - 1), "open")  # below threshold
        return recs

    if action == "escalate":
        for i in range(n_target):
            add(region, "silver" if i % 2 == 0 else "bronze", priority + (i % 2), "open")
        add(region, "gold", priority, "open")                       # blocked: tier
        add(region, "silver", priority, "open", "hold")             # blocked: note
        add("zz", "silver", 5, "open")                              # wrong region
        add(region, "silver", max(1, priority - 1), "open")         # below threshold
        add(region, "bronze", priority + 1, "escalated")            # already done
    else:
        for i in range(n_target):
            add(region, "silver" if i % 2 == 0 else "bronze", max(1, priority - (i % 2)),
                "escalated")
        add(region, "gold", priority, "escalated")                  # blocked: tier
        add(region, "bronze", priority, "escalated", "audit")       # blocked: note
        add("zz", "silver", 2, "escalated")                         # wrong region
        add(region, "silver", priority + 1, "escalated")            # above threshold
        add(region, "bronze", priority, "open")                     # not escalated
    return recs


def build_dataset(seed: int = 7, per_family: int = 4,
                  families: Sequence[str] = ("escalation", "closure")) -> List[BenchTask]:
    """Build a task list where each family repeats with fresh record tables.

    Repetition within a family is what makes memory measurable: the hidden rule is
    constant across a family, the concrete records are not.
    """
    tasks: List[BenchTask] = []
    for fi, fname in enumerate(families):
        family = FAMILIES[fname]
        for i in range(per_family):
            tseed = seed + fi * 1000 + i * 37
            region = ("eu", "us", "apac")[i % 3]
            priority = 3 if i % 2 == 0 else 4
            records = build_table(tseed, family.action, region, priority, fname)
            goal = family.goal_template.format(region=region, priority=priority)

            if family.action == "escalate":
                candidates = [r for r in records
                              if r.region == region and r.priority >= priority
                              and r.status == "open"]
            else:
                candidates = [r for r in records
                              if r.region == region and r.priority <= priority
                              and r.status == "escalated"]

            excluded = [r.rid for r in candidates if family.excluded(r)]
            expected = [r.rid for r in candidates if not family.excluded(r)]
            target_set = set(expected) | set(excluded)
            untouched = [r.rid for r in records if r.rid not in target_set]

            task = Task(
                goal=goal,
                task_id=f"{fname}_{i:02d}_{tseed}",
                context={"region": region, "priority": priority, "seed": tseed,
                         "action": family.action, "record_count": len(records)},
                tags=list(family.tags) + [fname, region],
                family=fname,
                max_steps=len(records) + 6,
            )
            tasks.append(BenchTask(
                task=task, family=fname, region=region, priority=priority,
                action=family.action, rules=list(family.rules),
                expected_ids=sorted(expected), excluded_ids=sorted(excluded),
                untouched_ids=sorted(untouched),
                initial_status={r.rid: r.status for r in records},
            ))
    return tasks


def task_env(bt: BenchTask) -> OpsEnv:
    """Rebuild the exact record table a task was generated against."""
    return OpsEnv(build_table(int(bt.task.context["seed"]), bt.action,
                              bt.region, bt.priority, bt.family))


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


def verify(bt: BenchTask, env: OpsEnv) -> Outcome:
    """Exact state comparison. This is the only source of truth for success."""
    target = bt.target_status
    checks: List[Dict[str, Any]] = []

    missing = [rid for rid in bt.expected_ids
               if env.records[rid].status != target]
    checks.append({
        "name": "targets_actioned", "passed": not missing,
        "detail": f"{len(bt.expected_ids) - len(missing)}/{len(bt.expected_ids)} done"
                  + (f"; missing {missing}" if missing else ""),
    })

    violated = [rid for rid in bt.excluded_ids
                if env.records[rid].status != bt.initial_status[rid]]
    if violated:
        # Name only the attribute that actually blocked the record. Listing every
        # field of the record teaches the wrong rule (a record excluded for its
        # note would also implicate its tier).
        parts = []
        for rid in violated:
            rec = env.records[rid]
            blocking = []
            for rule in bt.rules:
                if not rule.blocks(rec):
                    continue
                clause = f"{rule.field}={getattr(rec, rule.field)!r}"
                if rule.condition_field:
                    clause += (f" and {rule.condition_field}="
                               f"{getattr(rec, rule.condition_field)!r}")
                blocking.append(clause)
            parts.append(f"{rid}({' or '.join(blocking) if blocking else 'unknown'})")
        detail = "acted on records the rule excludes: " + ", ".join(parts)
    else:
        detail = "excluded records untouched"
    checks.append({
        "name": "hidden_rule_respected", "passed": not violated, "detail": detail,
    })

    collateral = [rid for rid in bt.untouched_ids
                  if env.records[rid].status != bt.initial_status[rid]]
    checks.append({
        "name": "no_collateral_changes", "passed": not collateral,
        "detail": "no unrelated records changed" if not collateral
                  else f"changed unrelated records: {collateral}",
    })

    checks.append({
        "name": "log_written", "passed": len(env.log) > 0,
        "detail": f"{len(env.log)} log entries",
    })

    passed = all(c["passed"] for c in checks)
    score = sum(1 for c in checks if c["passed"]) / len(checks)
    return Outcome(
        status=OutcomeStatus.SUCCESS if passed else OutcomeStatus.FAILURE,
        verified_by=VerifiedBy.PROGRAM,
        score=score,
        reason="all checks passed" if passed else
               "; ".join(c["detail"] for c in checks if not c["passed"]),
        checks=checks,
    )


def rules_brief(bt: BenchTask) -> str:
    """The human-readable rule set. Never shown to the agent; used in reports."""
    return "; ".join(r.describe() for r in bt.rules)


def dataset_summary(tasks: Sequence[BenchTask]) -> Dict[str, Any]:
    by_family: Dict[str, int] = {}
    for bt in tasks:
        by_family[bt.family] = by_family.get(bt.family, 0) + 1
    return {
        "n_tasks": len(tasks),
        "by_family": by_family,
        "rules": {f: rules_brief(next(t for t in tasks if t.family == f))
                  for f in by_family},
        "expected_actions": {f: FAMILIES[f].action for f in by_family},
    }
