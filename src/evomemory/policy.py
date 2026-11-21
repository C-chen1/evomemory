"""The offline agent policy.

A deterministic rule-follower, not a model. It exists so the harness can be
exercised end to end with zero API cost and bit-identical results.

It has two ways of honouring memory, matching the two arms of the ablation:

* **prose path** (modes B/C/D): read the injected text and look for values
  mentioned next to a negation cue. This is what a reader does with prose, and it
  is exactly where "gold *and* audit" degrades into "gold *or* audit".
* **executable path** (modes S/G/E): ask the harness to evaluate the stored
  predicate via the ``check_rule`` tool. The connective survives because it is
  never rendered into words.

Results from this policy validate the pipeline. They are **not** evidence about
model behaviour, and are labelled ``harness_selfcheck``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

_EXCLUDING = re.compile(
    r"(do not|don't|never|avoid|exclude[sd]?|excluding|skip|must not|not to|"
    r"without|leave|except|rather than)\b",
    re.IGNORECASE)


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", text) if s.strip()]


def blocked_values_from_guidance(guidance: str, candidates: List[str]) -> List[str]:
    """Infer which record values the guidance says to leave alone.

    Deliberately simple: a candidate value counts as blocked when it appears in
    the same sentence as a negation cue. Note that this cannot represent a
    conjunction — "gold and audit" marks both values blocked, which is the
    disjunction reading. That limitation is the reason the executable path exists.
    """
    blocked: List[str] = []
    for sentence in _sentences(guidance):
        if not _EXCLUDING.search(sentence):
            continue
        low = sentence.lower()
        for value in candidates:
            if re.search(rf"\b{re.escape(value)}\b", low) and value not in blocked:
                blocked.append(value)
    return blocked


class ScriptedOpsPolicy:
    """A stateful policy driven by the observation stream.

    Phases: init -> listing -> [checking] -> writing -> logging -> done
    """

    def __init__(self, guidance_getter=None):
        self.guidance_getter = guidance_getter
        self.reset()

    def reset(self) -> None:
        self.phase = "init"
        self.pending: List[str] = []
        self.to_skip: List[tuple[str, str]] = []
        self.to_check: List[str] = []
        self.candidates: List[str] = []
        self._last_task_id: Optional[str] = None

    # -- entry point -----------------------------------------------------
    def __call__(self, messages: List[Dict[str, str]]) -> str:
        system = messages[0]["content"] if messages else ""
        task = self._extract_task(system)
        task_id = str(task.get("task_id", ""))
        if task_id and task_id != self._last_task_id:
            self.reset()
            self._last_task_id = task_id
        guidance = self._extract_guidance(system)

        if self.phase == "init":
            self.phase = "listing"
            return json.dumps(self._list_action(task), ensure_ascii=False)

        if self.phase == "listing":
            return self._after_listing(messages, task, guidance)

        if self.phase == "checking":
            return self._after_check(messages, task)

        if self.phase == "writing":
            # A deliberate skip is an action, and it names the rule it follows.
            # Without this the harness can only see silence, which proves nothing.
            if self.to_skip:
                rid, exp_id = self.to_skip.pop(0)
                return json.dumps({"thought": f"{rid} is exempt under the learned rule",
                                   "tool": "skip_record",
                                   "args": {"rid": rid,
                                            "reason": "excluded by a learned rule",
                                            "because_experience": exp_id}},
                                  ensure_ascii=False)
            if self.pending:
                return self._next_write(task)
            self.phase = "logging"
            return self._log_action()

        if self.phase == "logging":
            self.phase = "done"
            return json.dumps({"thought": "finished", "tool": "finish", "args": {}},
                              ensure_ascii=False)

        return json.dumps({"thought": "already finished", "tool": "finish", "args": {}},
                          ensure_ascii=False)

    # -- phases ----------------------------------------------------------
    def _after_listing(self, messages, task, guidance) -> str:
        observation = self._last_observation(messages)
        records = observation.get("result") if isinstance(observation, dict) else None
        if not isinstance(records, list):
            records = []

        self.candidates = self._in_scope(records, task)

        if self._has_tool(task, "check_rule") and self.candidates:
            self.to_check = list(self.candidates)
            self.phase = "checking"
            return self._next_check()

        blocked = blocked_values_from_guidance(
            guidance, ["gold", "silver", "bronze", "hold", "audit"])
        by_id = {r["rid"]: r for r in records}
        self.pending = [rid for rid in self.candidates
                        if not self._blocked_by_text(by_id.get(rid, {}), blocked)]

        if self.pending:
            self.phase = "writing"
            return self._next_write(task)
        self.phase = "logging"
        return self._log_action(
            "no eligible records after applying known exclusions")

    def _after_check(self, messages, task) -> str:
        observation = self._last_observation(messages)
        result = observation.get("result") if isinstance(observation, dict) else None
        rows = []
        if isinstance(result, dict):
            rows = result.get("results") or ([result] if "rid" in result else [])
        for row in rows:
            # exempt=False -> no rule applies, act on it
            # exempt=True/None -> leave it alone (unknown is not false)
            if row.get("exempt") is False:
                self.pending.append(row["rid"])
            else:
                exempting = [v["experience_id"] for v in row.get("rules", [])
                             if v.get("verdict") == "true"]
                if exempting:
                    self.to_skip.append((row["rid"], exempting[0]))
        if self.to_check:
            return self._next_check()
        self.phase = "writing"
        if self.pending:
            return self._next_write(task)
        self.phase = "logging"
        return self._log_action("every candidate record was exempted by the learned rules")

    # -- actions ---------------------------------------------------------
    def _list_action(self, task: Dict[str, Any]) -> Dict[str, Any]:
        ctx = task.get("context", {}) or {}
        args: Dict[str, Any] = {"region": ctx.get("region", "any")}
        priority = ctx.get("priority")
        action = ctx.get("action", "escalate")
        if priority is not None:
            args["min_priority"] = priority if action == "escalate" else 1
        return {"thought": "enumerate the records in scope",
                "tool": "list_records", "args": args}

    def _next_check(self) -> str:
        rids = list(self.to_check)
        self.to_check = []
        return json.dumps({"thought": "check the learned rules against every candidate",
                           "tool": "check_rule", "args": {"rids": rids}},
                          ensure_ascii=False)

    def _next_write(self, task: Dict[str, Any]) -> str:
        rid = self.pending.pop(0)
        target = ("escalated" if (task.get("context", {}) or {}).get("action") == "escalate"
                  else "closed")
        return json.dumps({"thought": f"apply status to {rid}", "tool": "set_field",
                           "args": {"rid": rid, "field": "status", "value": target}},
                          ensure_ascii=False)

    def _log_action(self, text: str = "batch processed") -> str:
        return json.dumps({"thought": "log the batch", "tool": "log_entry",
                           "args": {"text": text}}, ensure_ascii=False)

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _extract_task(system: str) -> Dict[str, Any]:
        match = re.search(r"TASK:\s*(\{.*)", system, re.DOTALL)
        if not match:
            return {}
        blob = match.group(1)
        depth, end = 0, None
        for i, ch in enumerate(blob):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            return {}
        try:
            return json.loads(blob[:end])
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _extract_guidance(system: str) -> str:
        marker = "Reusable guidance from earlier tasks:"
        idx = system.find(marker)
        return system[idx:] if idx >= 0 else ""

    @staticmethod
    def _last_observation(messages: List[Dict[str, str]]) -> Any:
        for msg in reversed(messages):
            if msg["role"] == "user":
                try:
                    return json.loads(msg["content"])
                except json.JSONDecodeError:
                    return None
        return None

    @staticmethod
    def _has_tool(task: Dict[str, Any], name: str) -> bool:
        return any(isinstance(t, dict) and t.get("name") == name
                   for t in task.get("tools", []) or [])

    def _in_scope(self, records: List[Dict[str, Any]], task: Dict[str, Any]) -> List[str]:
        ctx = task.get("context", {}) or {}
        region = ctx.get("region", "any")
        priority = ctx.get("priority")
        action = ctx.get("action", "escalate")
        chosen: List[str] = []
        for rec in records:
            if region not in (None, "", "any") and rec.get("region") != region:
                continue
            if action == "escalate":
                if rec.get("status") != "open":
                    continue
                if priority is not None and rec.get("priority", 0) < priority:
                    continue
            else:
                if rec.get("status") != "escalated":
                    continue
                if priority is not None and rec.get("priority", 0) > priority:
                    continue
            chosen.append(rec["rid"])
        return chosen

    @staticmethod
    def _blocked_by_text(rec: Dict[str, Any], blocked: List[str]) -> bool:
        tier = str(rec.get("tier", "")).lower()
        note = str(rec.get("note", "")).lower()
        return tier in blocked or any(b in note for b in blocked)
