"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0

Ceilings on the work the condition engines do per template shape, so that a
change that makes a shape exponential or quadratic again fails here rather
than in a user's lint.
"""

from unittest import TestCase

from cfnlint.template import Template


def _eq(left, right):
    return {"Fn::Equals": [left, right]}


def _ref(name):
    return {"Ref": name}


def _cond(name):
    return {"Condition": name}


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
