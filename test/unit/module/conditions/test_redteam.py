"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0

Templates shaped like the red-team attacks on the condition engines (study
W1).  Each test builds its template from a small generator so the shape, not
a fixture file, is what is pinned down.
"""

from unittest import TestCase

from cfnlint.conditions._solver import PinSolver
from cfnlint.template import Template


def _eq(left, right):
    return {"Fn::Equals": [left, right]}


def _ref(name):
    return {"Ref": name}


def _cond(name):
    return {"Condition": name}


def _never(n):
    """False at every point, but undecided by three-valued evaluation until
    every parameter is assigned."""
    return {
        "Fn::Or": [
            {"Fn::And": [_cond(f"E{i}"), {"Fn::Not": [_cond(f"E{i}")]}]}
            for i in range(n)
        ]
    }


def refref_a8_big(n):
    """A0..A(n-1) in one Ref-vs-Ref class (Eq_i = A0 == A_i) and Lit = A0 == x;
    (1 + n) ** n points, above the point threshold from n = 7."""
    return {
        "Parameters": {f"A{i}": {"Type": "String"} for i in range(n)},
        "Conditions": {
            **{f"Eq{i}": _eq(_ref("A0"), _ref(f"A{i}")) for i in range(1, n)},
            "Lit": _eq(_ref("A0"), "x"),
        },
        "Resources": {
            "Q": {
                "Type": "AWS::SQS::Queue",
                "Condition": "Lit",
                "Properties": {"QueueName": {"Fn::If": ["Eq1", "same", "diff"]}},
            }
        },
    }


def refref_a8_budget(n):
    """n free parameters with Never, Needle true only where every P_i is not
    "a", Eq = A == B, Lit = A == x, Tie = Needle and Eq.  Tie and Lit hold
    together (all P_i other, A = B = x)."""
    conds = {f"E{i}": _eq(_ref(f"P{i}"), "a") for i in range(n)}
    conds["Never"] = _never(n)
    conds["AllOther"] = {"Fn::And": [{"Fn::Not": [_cond(f"E{i}")]} for i in range(n)]}
    conds["Needle"] = {"Fn::Or": [_cond("Never"), _cond("AllOther")]}
    conds["Eq"] = _eq(_ref("A"), _ref("B"))
    conds["Lit"] = _eq(_ref("A"), "x")
    conds["Tie"] = {"Fn::And": [_cond("Needle"), _cond("Eq")]}
    params = {f"P{i}": {"Type": "String"} for i in range(n)}
    params["A"] = {"Type": "String"}
    params["B"] = {"Type": "String"}
    return {
        "Parameters": params,
        "Conditions": conds,
        "Resources": {
            "Q": {
                "Type": "AWS::SQS::Queue",
                "Condition": "Tie",
                "Properties": {"QueueName": {"Fn::If": ["Lit", "same", "diff"]}},
            }
        },
    }


class TestRefVsRefExclusion(TestCase):
    """W1 A1/A2: a Ref-vs-Ref Fn::Equals is not exclusive with a literal
    Fn::Equals on the same parameter, so the CNF must not say NAND(P == Q,
    P == "x").  The SAT side answers from the CNF alone whenever a component
    is above the point threshold or a point search runs out of budget."""

    def _cnf_satisfiable(self, cfn, pins):
        solver = PinSolver(cfn.conditions._cnf, incremental=True)
        return solver.satisfiable(
            [(cfn.conditions._solver_params[name], value) for name, value in pins]
        )

    def test_literal_and_ref_equals_on_one_parameter_can_both_hold(self):
        """A0 = A1 = "x" makes Eq1 and Lit true together (refref_a8_big-7)."""
        cfn = Template("", refref_a8_big(7), ["us-east-1"])
        self.assertTrue(self._cnf_satisfiable(cfn, [("Lit", True), ("Eq1", True)]))
        self.assertTrue(
            cfn.conditions._satisfiable(
                [
                    (cfn.conditions._solver_params["Lit"], True),
                    (cfn.conditions._solver_params["Eq1"], True),
                ]
            )
        )
        self.assertEqual(
            list(cfn.conditions.build_scenarios({"Lit": {True}, "Eq1": {True, False}})),
            [{"Eq1": True, "Lit": True}, {"Eq1": False, "Lit": True}],
        )
        self.assertEqual(
            len(list(cfn.conditions.build_scenarios({"Lit": None, "Eq1": None}))), 4
        )
        self.assertFalse(cfn.conditions.check_implies({"Lit": True}, "Eq1"))
        # two Ref-vs-Ref equals on one parameter are not exclusive either
        self.assertTrue(self._cnf_satisfiable(cfn, [("Eq1", True), ("Eq2", True)]))

    def test_literal_equals_on_one_parameter_stay_exclusive(self):
        cfn = Template(
            "",
            {
                "Parameters": {"P": {"Type": "String"}, "Q": {"Type": "String"}},
                "Conditions": {
                    "IsA": _eq(_ref("P"), "a"),
                    "IsB": _eq(_ref("P"), "b"),
                    "Same": _eq(_ref("P"), _ref("Q")),
                },
            },
            ["us-east-1"],
        )
        self.assertFalse(self._cnf_satisfiable(cfn, [("IsA", True), ("IsB", True)]))
        self.assertTrue(self._cnf_satisfiable(cfn, [("IsA", True), ("Same", True)]))
        self.assertTrue(self._cnf_satisfiable(cfn, [("IsB", True), ("Same", True)]))
        self.assertEqual(
            list(cfn.conditions.build_scenarios({"IsA": None, "IsB": None})),
            [
                {"IsA": True, "IsB": False},
                {"IsA": False, "IsB": True},
                {"IsA": False, "IsB": False},
            ],
        )

    def test_budget_fallback_answers_from_a_sound_cnf(self):
        """The satisfying point of refref_a8_budget is the last one the point
        search visits; when the search runs out of budget the CNF answers,
        and it must allow Tie and Lit together (A = B = "x")."""
        cfn = Template("", refref_a8_budget(6), ["us-east-1"])
        self.assertTrue(self._cnf_satisfiable(cfn, [("Tie", True), ("Lit", True)]))
        self.assertEqual(
            list(cfn.conditions.build_scenarios({"Tie": {True}, "Lit": {True, False}})),
            [{"Lit": True, "Tie": True}, {"Lit": False, "Tie": True}],
        )
