"""The offline agent policy.

This is a *deterministic rule-follower*, not a model. It exists so that the
harness can be exercised end to end with zero API cost and bit-identical
results.

It matters that this is honest about what it measures: because the policy reads
natural-language guidance with a fixed rule, offline runs demonstrate that the
**plumbing** works (retrieval -> critique -> reconstruction -> injection ->
behaviour change -> write-back -> admission). They do **not** demonstrate that a
real LLM benefits from memory, and results produced with it are labelled
``harness_selfcheck`` and never reported as effectiveness evidence.
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


def blocked_values_from_guidance(guidance: str,
                                 candidates: List[str]) -> List[str]:
    """Infer which record values the guidance says to leave alone.

    Deliberately simple and general: a candidate value counts as blocked when it
    appears in the same sentence as a negation cue. This is how the policy
    "reads" memory; a real model would read it the way any reader would.
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
    """A stateful policy driven by the observation stream."""

    def __init__(self, guidance_getter=None):
        self.guidance_getter = guidance_getter
        self.reset()

    def reset(self) -> None:
        self.phase = "init"
        self.pending: List[str] = []
        self._last_task_id: Optional[str] = None

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        system = messages[0]["content"] if messages else ""
        task = self._extract_task(system)
        # Reset per task by identity, not by message count: the harness trims
        # history, so "len(messages) <= 2" is no longer a reliable new-task signal.
        task_id = str(task.get("task_id", ""))
        if task_id and task_id != self._last_task_id:
            self.reset()
            self._last_task_id = task_id
        guidance = self._extract_guidance(system)

        if self.phase == "init":
            self.phase = "listing"
            return json.dumps(self._list_action(task), ensure_ascii=False)

        if self.phase == "listing":
            observation = self._last_observation(messages)
            records = observation.get("result") if isinstance(observation, dict) else None
            if isinstance(records, list):
                self.pending = self._choose_targets(records, task, guidance)
            if self.pending:
                self.phase = "writing"
                return self._next_write(task)
            self.phase = "logging"
            return json.dumps({
                "thought": "nothing to change after applying guidance",
                "tool": "log_entry",
                "args": {"text": "no eligible records after applying known exclusions"},
            }, ensure_ascii=False)

        if self.phase == "writing":
            if self.pending:
                return self._next_write(task)
            self.phase = "logging"
            return json.dumps({"thought": "log the batch", "tool": "log_entry",
                               "args": {"text": "batch processed"}}, ensure_ascii=False)

        if self.phase == "logging":
            self.phase = "done"
            return json.dumps({"thought": "finished", "tool": "finish", "args": {}},
                              ensure_ascii=False)

        return json.dumps({"thought": "already finished", "tool": "finish", "args": {}},
                          ensure_ascii=False)

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _extract_task(system: str) -> Dict[str, Any]:
        match = re.search(r"TASK:\s*(\{.*)", system, re.DOTALL)
        if not match:
            return {}
        blob = match.group(1)
        # guidance (if any) was appended after the JSON object
        depth = 0
        end = None
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
    def _list_action(task: Dict[str, Any]) -> Dict[str, Any]:
        ctx = task.get("context", {}) or {}
        args = {"region": ctx.get("region", "any")}
        priority = ctx.get("priority")
        action = ctx.get("action", "escalate")
        if priority is not None:
            # the tool filters by minimum priority; closure needs the complement,
            # so the policy lists the region and filters locally
            args["min_priority"] = priority if action == "escalate" else 1
        return {"thought": "enumerate the records in scope",
                "tool": "list_records", "args": args}

    def _last_observation(self, messages: List[Dict[str, str]]) -> Any:
        for msg in reversed(messages):
            if msg["role"] == "user":
                try:
                    return json.loads(msg["content"])
                except json.JSONDecodeError:
                    return None
        return None

    def _choose_targets(self, records: List[Dict[str, Any]], task: Dict[str, Any],
                        guidance: str) -> List[str]:
        ctx = task.get("context", {}) or {}
        region = ctx.get("region", "any")
        priority = ctx.get("priority")
        action = ctx.get("action", "escalate")
        blocked = blocked_values_from_guidance(
            guidance, ["gold", "silver", "bronze", "hold", "audit"])

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
            if self._blocked(rec, blocked):
                continue
            chosen.append(rec["rid"])
        return chosen

    @staticmethod
    def _blocked(rec: Dict[str, Any], blocked: List[str]) -> bool:
        tier = str(rec.get("tier", "")).lower()
        note = str(rec.get("note", "")).lower()
        return tier in blocked or any(b in note for b in blocked)

    def _next_write(self, task: Dict[str, Any]) -> str:
        rid = self.pending.pop(0)
        target = "escalated" if task.get("context", {}).get("action") == "escalate" else "closed"
        return json.dumps({"thought": f"apply status to {rid}", "tool": "set_field",
                           "args": {"rid": rid, "field": "status", "value": target}},
                          ensure_ascii=False)
