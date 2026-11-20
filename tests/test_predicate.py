"""Predicate engine: logic, three-valued semantics, and adversarial input."""

import itertools

import pytest

from evomemory.predicate import (
    MAX_DEPTH, PredicateError, Tri, applicable, describe, evaluate,
    from_counterexamples, validate,
)

ALLOWED = {"tier", "note", "region", "priority", "status"}


def leaf(field, op, value=None):
    node = {"field": field, "op": op}
    if value is not None:
        node["value"] = value
    return node


# -- 1..5: the basic connectives -------------------------------------------


def test_single_condition():
    p = leaf("tier", "eq", "gold")
    assert evaluate(p, {"tier": "gold"}) is Tri.TRUE
    assert evaluate(p, {"tier": "silver"}) is Tri.FALSE


def test_conjunction_requires_both():
    p = {"all_of": [leaf("tier", "eq", "gold"), leaf("note", "eq", "audit")]}
    assert evaluate(p, {"tier": "gold", "note": "audit"}) is Tri.TRUE
    assert evaluate(p, {"tier": "gold", "note": ""}) is Tri.FALSE
    assert evaluate(p, {"tier": "silver", "note": "audit"}) is Tri.FALSE


def test_disjunction_requires_one():
    p = {"any_of": [leaf("tier", "eq", "gold"), leaf("note", "eq", "audit")]}
    assert evaluate(p, {"tier": "gold", "note": ""}) is Tri.TRUE
    assert evaluate(p, {"tier": "silver", "note": "audit"}) is Tri.TRUE
    assert evaluate(p, {"tier": "silver", "note": ""}) is Tri.FALSE


def test_negation():
    p = {"not": leaf("tier", "eq", "gold")}
    assert evaluate(p, {"tier": "silver"}) is Tri.TRUE
    assert evaluate(p, {"tier": "gold"}) is Tri.FALSE


def test_nested_combination():
    p = {
        "any_of": [
            {"all_of": [leaf("tier", "eq", "gold"), leaf("note", "eq", "audit")]},
            {"all_of": [leaf("region", "eq", "eu"), leaf("priority", "gte", 4)]},
        ]
    }
    assert evaluate(p, {"tier": "gold", "note": "audit", "region": "us", "priority": 1}) is Tri.TRUE
    assert evaluate(p, {"tier": "gold", "note": "", "region": "eu", "priority": 4}) is Tri.TRUE
    assert evaluate(p, {"tier": "gold", "note": "", "region": "eu", "priority": 3}) is Tri.FALSE


# -- 6..8: unknown and boundaries ------------------------------------------


def test_missing_field_is_unknown_not_false():
    """The bug this prevents: a rule silently stops applying when a field is absent."""
    p = leaf("tier", "eq", "gold")
    assert evaluate(p, {"note": "x"}) is Tri.UNKNOWN
    assert evaluate(p, {"note": "x"}) is not Tri.FALSE


def test_unknown_propagates_through_and_and_or():
    p_and = {"all_of": [leaf("tier", "eq", "gold"), leaf("note", "eq", "audit")]}
    assert evaluate(p_and, {"tier": "gold"}) is Tri.UNKNOWN      # note missing
    assert evaluate(p_and, {"tier": "silver"}) is Tri.FALSE      # already false
    p_or = {"any_of": [leaf("tier", "eq", "gold"), leaf("note", "eq", "audit")]}
    assert evaluate(p_or, {"tier": "silver"}) is Tri.UNKNOWN     # gold false, note unknown
    assert evaluate(p_or, {"tier": "gold"}) is Tri.TRUE          # already true
    assert evaluate({"not": leaf("tier", "eq", "gold")}, {}) is Tri.UNKNOWN


def test_type_mismatch_is_unknown():
    assert evaluate(leaf("priority", "gte", 3), {"priority": "high"}) is Tri.UNKNOWN
    assert evaluate(leaf("tier", "eq", 5), {"tier": "gold"}) is Tri.UNKNOWN


def test_boundary_values():
    assert evaluate(leaf("priority", "gte", 3), {"priority": 3}) is Tri.TRUE
    assert evaluate(leaf("priority", "gt", 3), {"priority": 3}) is Tri.FALSE
    assert evaluate(leaf("priority", "lte", 0), {"priority": 0}) is Tri.TRUE
    assert evaluate(leaf("priority", "lt", 0), {"priority": 0}) is Tri.FALSE


def test_set_membership_and_existence():
    assert evaluate(leaf("tier", "in", ["gold", "platinum"]), {"tier": "gold"}) is Tri.TRUE
    assert evaluate(leaf("tier", "in", ["gold"]), {"tier": "bronze"}) is Tri.FALSE
    assert evaluate(leaf("note", "exists"), {"note": ""}) is Tri.TRUE
    assert evaluate(leaf("note", "exists"), {}) is Tri.FALSE
    assert evaluate(leaf("note", "exists", False), {}) is Tri.TRUE


# -- 9: the decoy case that motivated all of this ---------------------------


def test_conjunction_decoy_discrimination():
    """gold+audit must match; gold-only and audit-only must not."""
    p = {"all_of": [leaf("tier", "eq", "gold"), leaf("note", "eq", "audit")]}
    assert evaluate(p, {"tier": "gold", "note": "audit"}) is Tri.TRUE
    assert evaluate(p, {"tier": "gold", "note": ""}) is Tri.FALSE
    assert evaluate(p, {"tier": "silver", "note": "audit"}) is Tri.FALSE
    # and the disjunction it was confused with behaves differently
    d = {"any_of": [leaf("tier", "eq", "gold"), leaf("note", "eq", "audit")]}
    assert evaluate(d, {"tier": "gold", "note": ""}) is Tri.TRUE   # would wrongly exempt


# -- 13: adversarial input -------------------------------------------------


@pytest.mark.parametrize("bad", [
    None, "all_of", 42, [], {},
    {"field": "tier"},                                  # missing op
    {"op": "eq", "value": "gold"},                      # missing field
    {"field": "tier", "op": "regex", "value": ".*"},    # unknown op
    {"field": "tier", "op": "eq"},                      # missing value
    {"field": "tier", "op": "in", "value": "gold"},     # in needs a list
    {"field": "priority", "op": "gt", "value": "3"},    # ordering needs a number
    {"all_of": []},                                     # empty group
    {"all_of": "nope"},
    {"all_of": [leaf("tier", "eq", "gold")], "any_of": [leaf("note", "eq", "x")]},
    {"field": "tier", "op": "eq", "value": "gold", "all_of": []},
    {"field": "__class__", "op": "eq", "value": 1},     # dunder access
    {"field": "a.b", "op": "eq", "value": 1},           # attribute path
])
def test_malformed_predicates_are_rejected(bad):
    with pytest.raises(PredicateError):
        validate(bad)


def test_field_whitelist_is_enforced():
    validate(leaf("tier", "eq", "gold"), ALLOWED)
    with pytest.raises(PredicateError):
        validate(leaf("password", "eq", "x"), ALLOWED)


def test_depth_and_size_limits():
    deep = leaf("tier", "eq", "gold")
    for _ in range(MAX_DEPTH + 2):
        deep = {"not": deep}
    with pytest.raises(PredicateError):
        validate(deep)
    wide = {"any_of": [leaf("tier", "eq", f"v{i}") for i in range(100)]}
    with pytest.raises(PredicateError):
        validate(wide)


def test_tri_is_not_truthy():
    """`if evaluate(...)` must fail loudly rather than silently treat unknown as truthy."""
    with pytest.raises(TypeError):
        bool(Tri.TRUE)


# -- equivalence with plain boolean logic (property test) -------------------


def test_and_or_agree_with_python_on_fully_defined_records():
    """Exhaustive over a small domain: no unknowns, so semantics must match."""
    values = ["gold", "silver", "bronze"]
    p_and = {"all_of": [leaf("tier", "eq", "gold"), leaf("note", "eq", "audit")]}
    p_or = {"any_of": [leaf("tier", "eq", "gold"), leaf("note", "eq", "audit")]}
    for tier, note in itertools.product(values, ["audit", "", "hold"]):
        rec = {"tier": tier, "note": note}
        expect_and = (tier == "gold") and (note == "audit")
        expect_or = (tier == "gold") or (note == "audit")
        assert (evaluate(p_and, rec) is Tri.TRUE) == expect_and
        assert (evaluate(p_or, rec) is Tri.TRUE) == expect_or


def test_not_is_an_involution():
    p = {"not": {"not": leaf("tier", "eq", "gold")}}
    for tier in ("gold", "silver"):
        assert evaluate(p, {"tier": tier}) is evaluate(leaf("tier", "eq", "gold"), {"tier": tier})


# -- connective inference from counterexamples -----------------------------


def test_infers_conjunction_from_one_record_with_two_attributes():
    p = from_counterexamples([[("tier", "gold"), ("note", "audit")]])
    assert p == {"all_of": [leaf("tier", "eq", "gold"), leaf("note", "eq", "audit")]}
    # the learned rule must reproduce the discriminating behaviour
    assert evaluate(p, {"tier": "gold", "note": "audit"}) is Tri.TRUE
    assert evaluate(p, {"tier": "gold", "note": ""}) is Tri.FALSE


def test_infers_disjunction_from_two_records_with_different_attributes():
    p = from_counterexamples([[("tier", "gold")], [("note", "hold")]])
    assert "any_of" in p
    assert evaluate(p, {"tier": "gold", "note": ""}) is Tri.TRUE
    assert evaluate(p, {"tier": "silver", "note": "hold"}) is Tri.TRUE
    assert evaluate(p, {"tier": "silver", "note": ""}) is Tri.FALSE


def test_single_attribute_violations_collapse_to_one_leaf():
    p = from_counterexamples([[("note", "hold")], [("note", "hold")]])
    assert p == leaf("note", "eq", "hold")


def test_no_evidence_returns_none():
    assert from_counterexamples([]) is None
    assert from_counterexamples([[], []]) is None


# -- applicability wrapper -------------------------------------------------


def test_applicable_wraps_validation_and_unknown():
    ok = applicable(leaf("tier", "eq", "gold"), {"tier": "gold"}, ALLOWED)
    assert ok.verdict is Tri.TRUE and ok.exempts
    missing = applicable(leaf("tier", "eq", "gold"), {}, ALLOWED)
    assert missing.verdict is Tri.UNKNOWN and not missing.exempts
    bad = applicable(leaf("secret", "eq", "x"), {}, ALLOWED)
    assert bad.verdict is Tri.UNKNOWN and "invalid" in bad.reason
    assert applicable(None, {}) .verdict is Tri.UNKNOWN


def test_describe_is_deterministic_and_readable():
    p = {"all_of": [leaf("tier", "eq", "gold"), leaf("note", "eq", "audit")]}
    assert describe(p) == "tier eq 'gold' and note eq 'audit'"
    assert describe(p) == describe(p)
