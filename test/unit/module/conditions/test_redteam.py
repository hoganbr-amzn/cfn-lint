"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0

Templates shaped like the red-team attacks on the condition engines (study
W1).  Each test builds its template from a small generator so the shape, not
a fixture file, is what is pinned down.
"""

from unittest import TestCase

from cfnlint.conditions import _points as points
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


def budget_fallback(n, prefix=""):
    """n free parameters, E_i = P_i == a, Never (false everywhere), H_j =
    Never or E_j; topic j on H_j with Fn::If on E_j, so (H_j true, E_j
    false) is an unsatisfiable question that a point search can only settle
    by visiting every point.  ``prefix`` makes an independent copy."""
    p = prefix
    conds = {f"{p}E{i}": _eq(_ref(f"{p}P{i}"), "a") for i in range(n)}
    conds[f"{p}Never"] = {
        "Fn::Or": [
            {"Fn::And": [_cond(f"{p}E{i}"), {"Fn::Not": [_cond(f"{p}E{i}")]}]}
            for i in range(n)
        ]
    }
    res = {}
    for j in range(n):
        conds[f"{p}H{j}"] = {"Fn::Or": [_cond(f"{p}Never"), _cond(f"{p}E{j}")]}
        res[f"{p}T{j}"] = {
            "Type": "AWS::SNS::Topic",
            "Condition": f"{p}H{j}",
            "Properties": {
                "TopicName": {"Fn::If": [f"{p}E{j}", f"{p}t{j}-a", f"{p}t{j}-b"]}
            },
        }
    return {
        "Parameters": {f"{p}P{i}": {"Type": "String"} for i in range(n)},
        "Conditions": conds,
        "Resources": res,
    }


def _merge(*templates):
    out = {"Parameters": {}, "Conditions": {}, "Resources": {}}
    for t in templates:
        for section in out:
            out[section].update(t.get(section, {}))
    return out


class TestPointBudget(TestCase):
    """W1 P1: the work budget bounds one point search; a component whose
    search runs out is answered by SAT for the rest of the lint, and after
    MAX_TRIPS such searches every component is."""

    def setUp(self):
        if points.ENGINE not in ("hybrid", "pp"):
            self.skipTest("point engine not selected")
        self._budget = points.STEP_BUDGET
        self._trips = points.MAX_TRIPS

    def tearDown(self):
        points.STEP_BUDGET = self._budget
        points.MAX_TRIPS = self._trips

    @staticmethod
    def _ask(cfn, pins):
        return cfn.conditions._solver.satisfiable(
            [(cfn.conditions._solver_params[n], v) for n, v in pins]
        )

    def test_component_that_trips_the_budget_is_routed_to_sat(self):
        points.STEP_BUDGET = 500
        cfn = Template("", budget_fallback(8), ["us-east-1"])
        solver = cfn.conditions._solver
        fallbacks = points._STATS["fallbacks"]
        self.assertFalse(self._ask(cfn, [("H0", True), ("E0", False)]))
        self.assertEqual(points._STATS["fallbacks"], fallbacks + 1)
        (component,) = solver._components
        self.assertEqual((component.route, component.why), ("sat", "budget"))
        # later distinct hard questions about the component do not search
        for j in range(1, 8):
            self.assertFalse(self._ask(cfn, [(f"H{j}", True), (f"E{j}", False)]))
            self.assertTrue(self._ask(cfn, [(f"H{j}", True), (f"E{j}", True)]))
        self.assertEqual(points._STATS["fallbacks"], fallbacks + 1)

    def test_budget_is_enough_for_an_easy_component(self):
        cfn = Template("", budget_fallback(8), ["us-east-1"])
        fallbacks = points._STATS["fallbacks"]
        for j in range(8):
            self.assertFalse(self._ask(cfn, [(f"H{j}", True), (f"E{j}", False)]))
        self.assertEqual(points._STATS["fallbacks"], fallbacks)
        (component,) = cfn.conditions._solver._components
        self.assertEqual(component.route, "pp")

    def test_after_max_trips_every_component_goes_to_sat(self):
        points.STEP_BUDGET = 500
        points.MAX_TRIPS = 1
        cfn = Template(
            "",
            _merge(budget_fallback(8, "X"), budget_fallback(8, "Y")),
            ["us-east-1"],
        )
        solver = cfn.conditions._solver
        fallbacks = points._STATS["fallbacks"]
        self.assertFalse(self._ask(cfn, [("XH0", True), ("XE0", False)]))
        self.assertFalse(self._ask(cfn, [("YH0", True), ("YE0", False)]))
        self.assertEqual(points._STATS["fallbacks"], fallbacks + 1)
        routes = sorted((c.route, c.why) for c in solver._components)
        self.assertEqual(routes, [("sat", "budget"), ("sat", "trips")])


def wide_eval(n):
    """One parameter Mode with n AllowedValues, Is_i = Mode == v_i, Any = Or
    of all Is_i; topic i on Is_i with Fn::If on Any.  Every evaluation of
    Any walks up to n disjuncts."""
    vals = [f"v{i:04d}" for i in range(n)]
    conds = {f"Is{i}": _eq(_ref("Mode"), vals[i]) for i in range(n)}
    conds["Any"] = {"Fn::Or": [_cond(f"Is{i}") for i in range(n)]}
    return {
        "Parameters": {
            "Mode": {"Type": "String", "Default": vals[0], "AllowedValues": vals}
        },
        "Conditions": conds,
        "Resources": {
            f"T{i}": {
                "Type": "AWS::SNS::Topic",
                "Condition": f"Is{i}",
                "Properties": {"TopicName": {"Fn::If": ["Any", f"t{i}-a", f"t{i}-b"]}},
            }
            for i in range(n)
        },
    }


class TestWorkIsNodeEvaluations(TestCase):
    """W1 P2: the budget counts node evaluations, so a step that walks a wide
    condition costs what it really costs, and a pinned-true "Ref == literal"
    fixes its reference instead of being searched for."""

    def setUp(self):
        if points.ENGINE not in ("hybrid", "pp"):
            self.skipTest("point engine not selected")
        self._budget = points.STEP_BUDGET

    def tearDown(self):
        points.STEP_BUDGET = self._budget

    @staticmethod
    def _ask(cfn, pins):
        return cfn.conditions._solver.satisfiable(
            [(cfn.conditions._solver_params[n], v) for n, v in pins]
        )

    def test_wide_or_charges_its_width(self):
        n = 64
        points.STEP_BUDGET = 300
        cfn = Template("", wide_eval(n), ["us-east-1"])
        solver = cfn.conditions._solver
        fallbacks = points._STATS["fallbacks"]
        # Any false and Is3 false: the search tries Mode's 64 values in
        # order and evaluates the 64-wide Or at each (short-circuiting at the
        # k-th disjunct for the k-th value), far more node evaluations than
        # steps; the budget trips and SAT gives the right answer
        self.assertFalse(self._ask(cfn, [("Any", False), ("Is3", False)]))
        self.assertEqual(points._STATS["fallbacks"], fallbacks + 1)
        self.assertGreater(solver._work, 300)

    def test_pinned_literal_equals_fixes_its_reference(self):
        n = 64
        cfn = Template("", wide_eval(n), ["us-east-1"])
        solver = cfn.conditions._solver
        self.assertTrue(self._ask(cfn, [("Is5", True), ("Any", True)]))
        before = solver._work
        self.assertFalse(self._ask(cfn, [("Is6", True), ("Any", False)]))
        self.assertTrue(self._ask(cfn, [("Is7", True), ("Is8", False)]))
        self.assertFalse(self._ask(cfn, [("Is7", True), ("Is8", True)]))
        # three questions over a 64-value parameter, each answered without
        # walking its values: a few node evaluations each, not 64 steps
        self.assertLess(solver._work - before, 3 * n)

    def test_literal_outside_allowed_values_is_impossible(self):
        template = wide_eval(4)
        template["Conditions"]["Other"] = _eq(_ref("Mode"), "nope")
        cfn = Template("", template, ["us-east-1"])
        self.assertFalse(self._ask(cfn, [("Other", True)]))
        self.assertTrue(self._ask(cfn, [("Other", False), ("Is1", True)]))


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
