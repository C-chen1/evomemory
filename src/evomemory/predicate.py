"""A small, total, three-valued predicate engine for executable memory.

Why this exists: an earlier experiment showed a model reading a learned rule
"exclude records with tier=gold **and** note=audit" as "exclude gold **or**
audit" and silently skipping records it should have acted on. Free text cannot
carry a logical connective reliably. So a rule that is meant to *act* is stored
as data and evaluated by code.

Design constraints, each of which is a test:

* **No ``eval``, no ``exec``, no attribute access.** A predicate can only read
  fields from a flat record through a fixed operator set. An LLM-produced rule can
  therefore never become code that touches the harness.
* **Three-valued, not two-valued.** A missing field or a type mismatch is
  ``UNKNOWN``, never ``FALSE``. Collapsing unknown into false is how a rule
  silently stops applying.
* **Total**: every input either validates or raises ``PredicateError``; every
  evaluation on a validated predicate terminates and returns a ``Tri``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class Tri(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    @property
    def is_true(self) -> bool:
        return self is Tri.TRUE

    def __bool__(self) -> bool:  # pragma: no cover - guard against `if tri:`
        raise TypeError(
            "Tri is three-valued: compare explicitly (is_true / is_false) "
            "instead of relying on truthiness")


class PredicateError(ValueError):
    """The predicate is malformed. Raised at validation time, never at eval time."""


COMPARISONS = {"eq", "ne", "in", "lt", "lte", "gt", "gte"}
OPERATORS = COMPARISONS | {"exists"}
GROUPS = {"all_of", "any_of", "not"}

MAX_DEPTH = 8
MAX_NODES = 64


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def validate(node: Any, allowed_fields: Optional[Iterable[str]] = None,
             max_depth: int = MAX_DEPTH, max_nodes: int = MAX_NODES) -> None:
    """Raise ``PredicateError`` if ``node`` is not a well-formed predicate."""
    allowed = set(allowed_fields) if allowed_fields is not None else None
    counter = {"n": 0}
    _validate(node, allowed, 0, max_depth, counter, max_nodes)


def _validate(node: Any, allowed: Optional[set], depth: int, max_depth: int,
              counter: Dict[str, int], max_nodes: int) -> None:
    if depth > max_depth:
        raise PredicateError(f"predicate nested deeper than {max_depth}")
    counter["n"] += 1
    if counter["n"] > max_nodes:
        raise PredicateError(f"predicate has more than {max_nodes} nodes")
    if not isinstance(node, dict):
        raise PredicateError(f"predicate node must be an object, got {type(node).__name__}")

    keys = set(node)
    group_keys = keys & GROUPS
    is_leaf = "field" in keys or "op" in keys

    if group_keys and is_leaf:
        raise PredicateError(f"node mixes group and leaf keys: {sorted(keys)}")
    if len(group_keys) > 1:
        raise PredicateError(f"node has more than one group key: {sorted(group_keys)}")
    if not group_keys and not is_leaf:
        raise PredicateError(f"node is neither a group nor a leaf: {sorted(keys)}")
    if is_leaf and not {"field", "op"} <= keys:
        raise PredicateError("leaf requires both 'field' and 'op'")

    if "all_of" in node or "any_of" in node:
        key = "all_of" if "all_of" in node else "any_of"
        children = node[key]
        if not isinstance(children, list) or not children:
            raise PredicateError(f"'{key}' must be a non-empty list")
        for child in children:
            _validate(child, allowed, depth + 1, max_depth, counter, max_nodes)
        return

    if "not" in node:
        _validate(node["not"], allowed, depth + 1, max_depth, counter, max_nodes)
        return

    field = node["field"]
    op = node["op"]
    if not isinstance(field, str) or not field:
        raise PredicateError("'field' must be a non-empty string")
    if field.startswith("__") or "." in field or "[" in field:
        raise PredicateError(f"illegal field name {field!r}")
    if allowed is not None and field not in allowed:
        raise PredicateError(f"field {field!r} is not in the allowed set {sorted(allowed)}")
    if op not in OPERATORS:
        raise PredicateError(f"unknown operator {op!r}; allowed: {sorted(OPERATORS)}")
    if op == "exists":
        if "value" in node and not isinstance(node["value"], bool):
            raise PredicateError("'exists' takes an optional boolean value")
        return
    if "value" not in node:
        raise PredicateError(f"operator {op!r} requires a 'value'")
    value = node["value"]
    if op == "in" and not isinstance(value, list):
        raise PredicateError("'in' requires a list value")
    if op in {"lt", "lte", "gt", "gte"} and not _is_number(value):
        raise PredicateError(f"operator {op!r} requires a numeric value")
    if not _is_scalar(value) and not isinstance(value, list):
        raise PredicateError("'value' must be a scalar or a list")


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def evaluate(node: Dict[str, Any], record: Mapping[str, Any]) -> Tri:
    """Evaluate a **validated** predicate against a flat record."""
    if "all_of" in node:
        saw_unknown = False
        for child in node["all_of"]:
            r = evaluate(child, record)
            if r is Tri.FALSE:
                return Tri.FALSE
            if r is Tri.UNKNOWN:
                saw_unknown = True
        return Tri.UNKNOWN if saw_unknown else Tri.TRUE

    if "any_of" in node:
        saw_unknown = False
        for child in node["any_of"]:
            r = evaluate(child, record)
            if r is Tri.TRUE:
                return Tri.TRUE
            if r is Tri.UNKNOWN:
                saw_unknown = True
        return Tri.UNKNOWN if saw_unknown else Tri.FALSE

    if "not" in node:
        r = evaluate(node["not"], record)
        if r is Tri.TRUE:
            return Tri.FALSE
        if r is Tri.FALSE:
            return Tri.TRUE
        return Tri.UNKNOWN

    field = node["field"]
    op = node["op"]
    present = field in record

    if op == "exists":
        want = node.get("value", True)
        return Tri.TRUE if present == want else Tri.FALSE

    if not present:
        return Tri.UNKNOWN
    actual = record[field]
    expected = node["value"]

    if op == "eq":
        return _tri_from_bool(_equal(actual, expected))
    if op == "ne":
        return _tri_from_bool(_equal(actual, expected), invert=True)
    if op == "in":
        outcomes = [_equal(actual, candidate) for candidate in expected]
        if any(o is True for o in outcomes):
            return Tri.TRUE
        if any(o is None for o in outcomes):
            return Tri.UNKNOWN
        return Tri.FALSE
    # ordering operators
    if not (_is_number(actual) and _is_number(expected)):
        return Tri.UNKNOWN
    a, b = float(actual), float(expected)
    if op == "lt":
        return _tri_from_bool(a < b)
    if op == "lte":
        return _tri_from_bool(a <= b)
    if op == "gt":
        return _tri_from_bool(a > b)
    if op == "gte":
        return _tri_from_bool(a >= b)
    raise PredicateError(f"unhandled operator {op!r}")  # pragma: no cover


def _equal(actual: Any, expected: Any) -> Optional[bool]:
    """``None`` means the comparison is not decidable (type mismatch)."""
    if actual is None or expected is None:
        return actual is expected
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if _is_number(actual) and _is_number(expected):
        return float(actual) == float(expected)
    if isinstance(actual, str) and isinstance(expected, str):
        return actual == expected
    return None


def _tri_from_bool(value: Optional[bool], invert: bool = False) -> Tri:
    if value is None:
        return Tri.UNKNOWN
    if invert:
        value = not value
    return Tri.TRUE if value else Tri.FALSE


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def describe(node: Dict[str, Any]) -> str:
    """Deterministic natural-language rendering. No model involved."""
    if "all_of" in node:
        return " and ".join(_paren(describe(c)) for c in node["all_of"])
    if "any_of" in node:
        return " or ".join(_paren(describe(c)) for c in node["any_of"])
    if "not" in node:
        return f"not ({describe(node['not'])})"
    op, field, value = node["op"], node["field"], node.get("value")
    if op == "exists":
        return f"{field} is {'present' if value else 'absent'}"
    if op == "in":
        rendered = ", ".join(repr(v) for v in value)
        return f"{field} in [{rendered}]"
    return f"{field} {op} {value!r}"


def _paren(text: str) -> str:
    return f"({text})" if (" and " in text or " or " in text) else text


# --------------------------------------------------------------------------
# learning a predicate from counterexamples
# --------------------------------------------------------------------------


def from_counterexamples(violations: Sequence[Sequence[Tuple[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Infer the connective from *how many* attributes each violation carried.

    This is the part that fixes the observed failure. The evidence distinguishes
    the two cases that prose cannot:

    * one record violating two attributes at once -> the rule is a conjunction
      over those attributes (the record had both, so both are required);
    * several records, each violating a different single attribute -> the rule is
      a disjunction (any one of them is enough to be exempt).

    Returns ``None`` when there is no evidence at all.
    """
    cleaned: List[List[Tuple[str, Any]]] = []
    for record_attrs in violations:
        attrs = [(f, v) for f, v in record_attrs if isinstance(f, str) and f]
        if attrs:
            cleaned.append(attrs)
    if not cleaned:
        return None

    multi = [attrs for attrs in cleaned if len(attrs) >= 2]
    if multi:
        # conjunction: take the attributes that the *most* records agree on,
        # falling back to the evidence of the record with the most attributes
        best = max(multi, key=len)
        return _all_of(best)

    # every violation was a single attribute
    distinct: List[Tuple[str, Any]] = []
    for attrs in cleaned:
        if attrs[0] not in distinct:
            distinct.append(attrs[0])
    if len(distinct) == 1:
        return _leaf(*distinct[0])
    return {"any_of": [_leaf(f, v) for f, v in distinct]}


def _leaf(field: str, value: Any) -> Dict[str, Any]:
    return {"field": field, "op": "eq", "value": value}


def _all_of(attrs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    if len(attrs) == 1:
        return _leaf(*attrs[0])
    return {"all_of": [_leaf(f, v) for f, v in attrs]}


# --------------------------------------------------------------------------
# applicability: combining a predicate with a task scope
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Applicability:
    verdict: Tri
    reason: str

    @property
    def exempts(self) -> bool:
        return self.verdict is Tri.TRUE


def applicable(condition: Optional[Dict[str, Any]], record: Mapping[str, Any],
               allowed_fields: Optional[Iterable[str]] = None) -> Applicability:
    """Does the stored condition match this record?

    Nothing here is allowed to *act*; the result is data for the harness to
    decide with. An ``UNKNOWN`` means the rule could not be decided on the
    available fields, which the caller must not treat as "no match".
    """
    if not condition:
        return Applicability(Tri.UNKNOWN, "no structured condition stored")
    try:
        validate(condition, allowed_fields)
    except PredicateError as exc:
        return Applicability(Tri.UNKNOWN, f"invalid predicate: {exc}")
    verdict = evaluate(condition, record)
    if verdict is Tri.UNKNOWN:
        return Applicability(Tri.UNKNOWN, "required field missing or type mismatch")
    return Applicability(verdict, "matched" if verdict is Tri.TRUE else "not matched")
