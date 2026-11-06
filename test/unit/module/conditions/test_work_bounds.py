"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0

Ceilings on the work the condition engines do per template shape, at two
sizes each, so that a change that makes a shape quadratic again fails here
rather than in a user's lint.
"""

import math
from test.unit.module.conditions.test_points import (
    _cond,
    _eq,
    _ref,
    cond_chain_narrow,
    wide_eval,
)
from unittest import TestCase

from cfnlint.template import Template


def many_equals(n):
    """n Fn::Equals comparing one parameter to n different literals."""
    return {
        "Parameters": {"P": {"Type": "String"}},
        "Conditions": {f"Is{i}": _eq(_ref("P"), f"v{i}") for i in range(n)},
        "Resources": {},
    }


def doubly_referencing_chain(n):
    """C0 is an Fn::Equals and each later condition references the one
    before it twice, so expanding every reference would build a tree of
    2 ** n nodes."""
    conditions = {"C0": _eq(_ref("P"), "a")}
    for i in range(1, n):
        conditions[f"C{i}"] = {
            "Fn::Or": [
                _cond(f"C{i - 1}"),
                {"Fn::And": [_cond(f"C{i - 1}"), _eq(_ref(f"Q{i}"), "b")]},
            ]
        }
    return {
        "Parameters": {
            "P": {"Type": "String"},
            **{f"Q{i}": {"Type": "String"} for i in range(1, n)},
        },
        "Conditions": conditions,
        "Resources": {},
    }


class TestWorkBounds(TestCase):
    def test_referenced_condition_is_expanded_once(self):
        """A condition referenced twice by the next one, forty deep, is built
        once per name in both engines, not once per path to it."""
        n = 40
        cfn = Template("", doubly_referencing_chain(n), ["us-east-1"])
        self.assertEqual(len(cfn.conditions._conditions), n)
        self.assertEqual(len(cfn.conditions._memo.named), n)
        last = f"C{n - 1}"
        self.assertEqual(len(cfn.conditions.get(last).equals), n)
        context = cfn.context.conditions
        self.assertEqual(len(context.conditions), n)
        self.assertEqual(len(context.conditions[last].equals), n)
        self.assertTrue(cfn.conditions.check_implies({"C0": True}, last))
        self.assertFalse(context.evolve({last: False}).is_reachable("C0", True))

    def test_exclusion_group_clauses_are_n_log_n(self):
        """The mutual exclusion of n Fn::Equals on one parameter costs
        n log n clauses, not n squared, in both engines."""
        for n in (64, 256):
            with self.subTest(n=n):
                cfn = Template("", many_equals(n), ["us-east-1"])
                ceiling = 2 * n * math.ceil(math.log2(n))
                self.assertLessEqual(len(cfn.conditions._cnf.data), ceiling)
                self.assertLessEqual(
                    len(cfn.context.conditions._base_cnf.data), ceiling
                )

    def test_wide_or_costs_its_width_per_question(self):
        """A question about an Or of n conditions over one parameter costs a
        number of node evaluations proportional to n."""
        for n in (64, 256):
            with self.subTest(n=n):
                cfn = Template("", wide_eval(n), ["us-east-1"])
                solver = cfn.conditions._solver
                for i in range(n):
                    cfn.conditions.check_implies({f"Is{i}": True}, "Any")
                    list(cfn.conditions.build_scenarios({f"Is{i}": None, "Any": None}))
                self.assertLessEqual(solver._work, 4 * n * n)

    def test_condition_chain_costs_its_length(self):
        """Questions about a chain of n conditions referencing each other
        cost a number of node evaluations proportional to n."""
        for n in (200, 1000):
            with self.subTest(n=n):
                cfn = Template("", cond_chain_narrow(n), ["us-east-1"])
                solver = cfn.conditions._solver
                last = f"C{n - 1}"
                cfn.conditions.check_implies({last: True}, "C0")
                list(cfn.conditions.build_scenarios({last: {True}, "C0": None}))
                self.assertLessEqual(solver._work, 1500 * n)

    def test_free_parameters_are_one_component_each(self):
        """Conditions on unrelated parameters are solved apart: a question
        about one touches one component, however many there are."""
        n = 50
        cfn = Template(
            "",
            {
                "Parameters": {f"P{i}": {"Type": "String"} for i in range(n)},
                "Conditions": {
                    **{f"E{i}": _eq(_ref(f"P{i}"), "a") for i in range(n)},
                    "Both": {"Fn::And": [_cond("E0"), _cond("E1")]},
                },
                "Resources": {},
            },
            ["us-east-1"],
        )
        solver = cfn.conditions._solver
        self.assertTrue(cfn.conditions.check_implies({"Both": True}, "E0"))
        self.assertEqual(len(solver._components), n - 1)
        self.assertLess(solver._work, 20 * n)
