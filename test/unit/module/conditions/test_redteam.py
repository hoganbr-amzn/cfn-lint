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


def proj_subsets(n, m=12):
    """m free parameters (2 ** m points, one component); W_i reads every
    parameter; n queues each set 8 Fn::If properties on a distinct 8-subset
    of the W_i, so every queue is a new scenario question."""
    import itertools

    conds = {f"E{i}": _eq(_ref(f"P{i}"), "a") for i in range(m)}
    for i in range(m):
        terms = [_cond(f"E{i}")]
        for d in range(1, m):
            j, k = (i + d) % m, (i + d + 1) % m
            terms.append({"Fn::And": [_cond(f"E{j}"), {"Fn::Not": [_cond(f"E{k}")]}]})
        conds[f"W{i}"] = {"Fn::Or": terms}
    props = [
        "DelaySeconds",
        "MaximumMessageSize",
        "MessageRetentionPeriod",
        "ReceiveMessageWaitTimeSeconds",
        "VisibilityTimeout",
        "KmsDataKeyReusePeriodSeconds",
        "SqsManagedSseEnabled",
        "QueueName",
    ]
    subsets = list(itertools.combinations(range(m), 8))
    res = {}
    for q in range(n):
        res[f"Queue{q}"] = {
            "Type": "AWS::SQS::Queue",
            "Properties": {
                name: {"Fn::If": [f"W{wi}", f"q{q}-{name}-a", f"q{q}-{name}-b"]}
                for name, wi in zip(props, subsets[q % len(subsets)])
            },
        }
    return {
        "Parameters": {f"P{i}": {"Type": "String"} for i in range(m)},
        "Conditions": conds,
        "Resources": res,
    }


class TestScenarioProjection(TestCase):
    """W1 P3: scenario enumeration over a small component answers from one
    truth table per component instead of walking its points once per
    distinct question; a large component's per-question projection is
    capped and then goes prefix by prefix."""

    def setUp(self):
        if points.ENGINE not in ("hybrid", "pp"):
            self.skipTest("point engine not selected")
        self._proj = points.PROJ_BUDGET

    def tearDown(self):
        points.PROJ_BUDGET = self._proj

    @staticmethod
    def _questions(cfn, n):
        import itertools

        m = sum(1 for name in cfn.template["Conditions"] if name.startswith("W"))
        subsets = list(itertools.combinations(range(m), 8))
        for q in range(n):
            yield [f"W{i}" for i in subsets[q % len(subsets)]]

    def _check_against_cnf(self, cfn, n):
        legacy = cfn.conditions
        reference = PinSolver(legacy._cnf, incremental=True)
        for names in self._questions(cfn, n):
            free = [legacy._solver_params[x] for x in names]
            for pins in ([], [(legacy._solver_params["E0"], True)]):
                got = legacy._solver.models(pins, free, 128)
                want = reference.models(pins, free, 128)
                self.assertEqual(got, want, names)

    def test_small_component_answers_from_one_table(self):
        cfn = Template("", proj_subsets(40), ["us-east-1"])
        solver = cfn.conditions._solver
        self._check_against_cnf(cfn, 40)
        (component,) = solver._components
        self.assertEqual(component.points, 4096)
        self.assertIsNotNone(component.columns)
        # every W_i has a column by now; further distinct questions and
        # satisfiability checks evaluate no node at all
        work = solver._work
        for names in self._questions(cfn, 495):
            syms = [cfn.conditions._solver_params[x] for x in names]
            solver.models([], syms, 128)
            solver.satisfiable([(syms[0], True), (syms[1], False)])
        self.assertEqual(solver._work, work)

    def test_large_component_projection_is_capped(self):
        points.PROJ_BUDGET = 2000
        template = proj_subsets(30)
        # a 13th parameter joined to the component doubles its points past
        # ENUM_MAX while every question still reads the first twelve
        template["Parameters"]["P12"] = {"Type": "String"}
        template["Conditions"]["E12"] = _eq(_ref("P12"), "a")
        template["Conditions"]["X"] = {"Fn::And": [_cond("E12"), _cond("W0")]}
        cfn = Template("", template, ["us-east-1"])
        solver = cfn.conditions._solver
        self.assertTrue(solver.satisfiable([]))  # builds the components
        (component,) = solver._components
        self.assertGreater(component.points, points.ENUM_MAX)
        self._check_against_cnf(cfn, 30)
        self.assertIsNone(component.columns)
        self.assertGreater(component.projected, 2000)
        # the cap was reached, so later questions walk prefixes with the
        # component's own search and projection work stays where it was
        projected = component.projected
        self._check_against_cnf(cfn, 30)
        self.assertEqual(component.projected, projected)


def cond_chain_narrow(n, k=8):
    """n conditions chained through Condition: over k parameters (routed to
    points): C_i = And(C_{i-1}, P == a) or Or(C_{i-1}, P == b) alternately."""
    conds = {"C0": _eq(_ref("P0"), "a")}
    for i in range(1, n):
        if i % 2:
            conds[f"C{i}"] = {"Fn::And": [_cond(f"C{i-1}"), _eq(_ref(f"P{i % k}"), "a")]}
        else:
            conds[f"C{i}"] = {"Fn::Or": [_cond(f"C{i-1}"), _eq(_ref(f"P{i % k}"), "b")]}
    return {
        "Parameters": {f"P{i}": {"Type": "String"} for i in range(k)},
        "Conditions": conds,
        "Resources": {
            "T": {
                "Type": "AWS::SNS::Topic",
                "Condition": f"C{n - 1}",
                "Properties": {"TopicName": {"Fn::If": ["C0", "a", "b"]}},
            }
        },
    }


class TestLongConditionChain(TestCase):
    """W1 P4: evaluating a Condition: chain longer than the interpreter's
    recursion limit must not raise, in the point engine or in dependency
    analysis."""

    def test_chain_of_two_thousand_conditions_is_answered(self):
        n = 2000
        cfn = Template("", cond_chain_narrow(n), ["us-east-1"])
        legacy = cfn.conditions
        last = f"C{n - 1}"
        reference = PinSolver(legacy._cnf, incremental=True)
        pins = [(legacy._solver_params[last], True)]
        free = [legacy._solver_params["C0"]]
        self.assertEqual(
            legacy._solver.models(pins, free, 128), reference.models(pins, free, 128)
        )
        self.assertEqual(
            list(legacy.build_scenarios({last: {True}, "C0": {True, False}})),
            [{"C0": True, last: True}, {"C0": False, last: True}],
        )
        self.assertFalse(legacy.check_implies({last: True}, "C0"))
        self.assertTrue(legacy.check_implies({"C0": False}, f"C{n - 2}") is False)
        # the context engine's questions walk the same chain
        ctx = cfn.context.conditions.evolve({last: True})
        self.assertTrue(ctx.is_reachable("C0", True))
        self.assertTrue(ctx.is_reachable("C0", False))

    def test_condition_reference_cycle_is_still_detected(self):
        if points.ENGINE not in ("hybrid", "pp"):
            self.skipTest("point engine not selected")
        cfn = Template(
            "",
            {
                "Parameters": {"P": {"Type": "String"}},
                "Conditions": {
                    "E": _eq(_ref("P"), "a"),
                    "A": {"Fn::Or": [_cond("B"), _cond("E")]},
                    "B": {"Fn::And": [_cond("A"), {"Fn::Not": [_cond("E")]}]},
                },
            },
            ["us-east-1"],
        )
        # the legacy engine drops a cyclic condition when it builds it; the
        # context engine keeps it and its solver must route it to SAT
        self.assertEqual(sorted(cfn.conditions._conditions), ["E"])
        solver = cfn.context.conditions._solver
        self.assertTrue(solver.satisfiable([]))
        routes = sorted((c.route, c.why) for c in solver._components)
        self.assertEqual(routes, [("pp", ""), ("sat", "cycle")])
        self.assertEqual(len(solver._cyclic), 2)


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
