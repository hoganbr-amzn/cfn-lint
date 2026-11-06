"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

import itertools
import json
from unittest.mock import patch

import pytest
from sympy import Not, Symbol
from sympy.logic.inference import satisfiable

from cfnlint.api import ManualArgs, lint
from cfnlint.context import create_context_for_template
from cfnlint.context.conditions._condition import Condition
from cfnlint.context.conditions._conditions import Conditions, _Pinned
from cfnlint.context.conditions.exceptions import Unsatisfiable
from cfnlint.template import Template


def template():
    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Parameters": {
            "Environment": {
                "Type": "String",
                "Default": "dev",
                "AllowedValues": ["dev", "prod"],
            },
            "Name": {
                "Type": "String",
                "Default": "bi",
                "AllowedValues": ["bi", "ai"],
            },
            "AmiId": {
                "Type": "String",
                "Default": "",
            },
        },
        "Conditions": {
            "IsUsEast1": {"Fn::Equals": [{"Ref": "AWS::Region"}, "us-east-1"]},
            "IsNotUsEast1": {"Fn::Not": [{"Condition": "IsUsEast1"}]},
            "IsUsWest2": {"Fn::Equals": [{"Ref": "AWS::Region"}, "us-west-2"]},
            "IsProd": {"Fn::Equals": [{"Ref": "Environment"}, "prod"]},
            "IsDev": {"Fn::Equals": [{"Ref": "Environment"}, "dev"]},
            "IsAmiProvided": {"Fn::Not": [{"Fn::Equals": [{"Ref": "AmiId"}, ""]}]},
            "IsAi": {"Fn::Equals": [{"Ref": "Name"}, "ai"]},
            "IsBi": {"Fn::Equals": [{"Ref": "Name"}, "bi"]},
            "IsUsEast1OrUsWest2": {
                "Fn::Or": [
                    {"Condition": "IsUsEast1"},
                    {"Condition": "IsUsWest2"},
                ]
            },
            "IsUsEast1AndProd": {
                "Fn::And": [
                    {"Condition": "IsUsEast1"},
                    {"Condition": "IsProd"},
                ]
            },
            "IsStaticTrue": {"Fn::Equals": [1, "1"]},
            "IsStaticFalse": {"Fn::Equals": [1, "2"]},
            "TwoObjects": {
                "Fn::Equals": [
                    {"Ref": "Environment"},
                    {"Ref": "AWS::Region"},
                ]
            },
            "TwoObjectsSameHash": {
                "Fn::Equals": [
                    {"Ref": "AWS::Region"},
                    {"Ref": "Environment"},
                ]
            },
            "BadAwsFn": {"Ref": "Foo"},
            "BadFn": {"Key": "Foo"},
            "BadNotList": {"Fn::And": "Foo"},
            "UseBadFn": {"Condition": "BadFn"},
            "UseBadCondition": {"Condition": []},
            "EqualsTooLong": {"Fn::Equals": ["a", "b", "c"]},
            "EqualsWrongTypes": {"Fn::Equals": ["a", []]},
            "NotWithTwoItems": {
                "Fn::Not": [
                    {"Condition": "IsAi"},
                    {"Condition": "IsBi"},
                ]
            },
        },
    }


def test_conditions():
    cfn = Template(None, template(), regions=["us-east-1"])
    context = create_context_for_template(cfn)

    assert len(context.conditions.conditions) == 22
    assert context.conditions.conditions["IsStaticTrue"].fn_equals.is_static is True
    assert context.conditions.conditions["IsStaticFalse"].fn_equals.is_static is False
    assert (
        context.conditions.conditions["TwoObjects"].fn_equals.hash
        == context.conditions.conditions["TwoObjectsSameHash"].fn_equals.hash
    )
    assert context.conditions.conditions["TwoObjects"].fn_equals.is_static is None

    for k in [
        "IsUsEast1",
        "IsNotUsEast1",
        "IsUsWest2",
        "IsUsEast1OrUsWest2",
        "IsUsEast1AndProd",
    ]:
        assert context.conditions.conditions[k].is_region is True

    for k in ["IsAmiProvided", "IsProd", "IsDev", "IsAi", "IsBi"]:
        assert context.conditions.conditions[k].is_region is False

    for k in [
        "BadAwsFn",
        "BadFn",
        "BadNotList",
        "UseBadCondition",
        "EqualsTooLong",
        "EqualsWrongTypes",
        "NotWithTwoItems",
    ]:
        assert context.conditions.conditions[k].fn_equals.left.instance is None
        assert context.conditions.conditions[k].fn_equals.right.instance is None
        assert context.conditions.conditions[k].fn_equals.is_static is None
        assert context.conditions.conditions[k].is_region is False

    assert (
        context.conditions.conditions["UseBadFn"].condition.fn_equals.left.instance
        is None
    )
    assert (
        context.conditions.conditions["UseBadFn"].condition.fn_equals.right.instance
        is None
    )
    assert (
        context.conditions.conditions["UseBadFn"].condition.fn_equals.is_static is None
    )


@pytest.mark.parametrize(
    "current_status,new_status,expected",
    [
        ({}, {}, {}),
        ({"IsUsEast1": True}, {}, {"IsUsEast1": True}),
        (
            {"IsUsEast1": True},
            {"IsUsEast1": False},
            Unsatisfiable(
                new_status={"IsUsEast1": False}, current_status={"IsUsEast1": True}
            ),
        ),
        (
            {"IsUsEast1": True},
            {"IsNotUsEast1": True},
            Unsatisfiable(
                new_status={"IsNotUsEast1": True},
                current_status={"IsUsEast1": True},
            ),
        ),
        (
            {"IsUsEast1": False, "IsUsWest2": False},
            {"IsUsEast1OrUsWest2": True},
            Unsatisfiable(
                new_status={"IsUsEast1OrUsWest2": True},
                current_status={"IsUsEast1": True},
            ),
        ),
        (
            {"IsDev": False},
            {"IsProd": False},
            Unsatisfiable(new_status={"IsProd": False}, current_status={"IsDev": True}),
        ),
        ({}, {"BadFn": True}, {"BadFn": True}),
        ({}, {"BadFn": False}, {"BadFn": False}),
    ],
)
def test_condition_status(current_status, new_status, expected):
    cfn = Template(None, template(), regions=["us-east-1"])
    context = create_context_for_template(cfn)

    context = context.evolve(conditions=context.conditions.evolve(current_status))

    if isinstance(expected, Exception):
        with pytest.raises(ValueError):
            context.conditions.evolve(new_status)
    else:
        context = context.evolve(conditions=context.conditions.evolve(new_status))
        assert context.conditions.status == expected


@pytest.mark.parametrize(
    "current_status,instance,expected",
    [
        ({}, {"Foo": "Bar"}, [({"Foo": "Bar"}, {})]),
        (
            {},
            {"Fn::If": ["IsUsEast1", {"Foo": "Foo"}, {"Bar": "Bar"}]},
            [
                ({"Foo": "Foo"}, {"IsUsEast1": True}),
                ({"Bar": "Bar"}, {"IsUsEast1": False}),
            ],
        ),
        (
            {
                "IsUsEast1": True,
            },
            {"Fn::If": ["IsUsEast1", {"Foo": "Foo"}, {"Bar": "Bar"}]},
            [
                ({"Foo": "Foo"}, {"IsUsEast1": True}),
            ],
        ),
        (
            {
                "IsUsEast1": False,
            },
            {"Fn::If": ["IsUsEast1", {"Foo": "Foo"}, {"Bar": "Bar"}]},
            [
                ({"Bar": "Bar"}, {"IsUsEast1": False}),
            ],
        ),
        (
            {},
            {"Ref": "AWS::NoValue"},
            [
                (None, {}),
            ],
        ),
        (
            {},
            [{"Foo": {"Fn::If": ["IsUsEast1", "Foo", "Bar"]}}],
            [
                ([{"Foo": {"Fn::If": ["IsUsEast1", "Foo", "Bar"]}}], {}),
            ],
        ),
        (
            {},
            [{"Fn::If": ["IsUsEast1", {"Foo": "Bar"}, {"Ref": "AWS::NoValue"}]}],
            [
                ([{"Foo": "Bar"}], {"IsUsEast1": True}),
                ([], {"IsUsEast1": False}),
            ],
        ),
        (
            {"IsUsEast1": True, "IsProd": True},
            {
                "A": {"Fn::If": ["IsUsEast1AndProd", 1, 2]},
                "B": {"Fn::If": ["IsAi", 10, 11]},
            },
            [
                (
                    {"A": 1, "B": 10},
                    {
                        "IsUsEast1": True,
                        "IsProd": True,
                        "IsUsEast1AndProd": True,
                        "IsAi": True,
                    },
                ),
                (
                    {"A": 1, "B": 11},
                    {
                        "IsUsEast1": True,
                        "IsProd": True,
                        "IsUsEast1AndProd": True,
                        "IsAi": False,
                    },
                ),
            ],
        ),
        (
            {},
            {
                "A": {"Fn::If": ["IsUsEast1AndProd", 1]},
            },
            [
                (
                    {"A": {"Fn::If": ["IsUsEast1AndProd", 1]}},
                    {},
                ),
            ],
        ),
        (
            {},
            [
                "a",
                "b",
                {"Fn::If": ["IsDev", "c", {"Ref": "AWS::NoValue"}]},
                {"Fn::If": ["IsProd", "d", {"Ref": "AWS::NoValue"}]},
            ],
            [
                (["a", "b", "c"], {"IsDev": True, "IsProd": False}),
                (["a", "b", "d"], {"IsDev": False, "IsProd": True}),
            ],
        ),
    ],
)
def test_evolve_from_instance(current_status, instance, expected):
    cfn = Template(None, template(), regions=["us-east-1"])
    context = create_context_for_template(cfn)

    context = context.evolve(
        conditions=context.conditions.evolve(current_status),
        functions=["Fn::If", "Ref"],
    )

    results = list(context.conditions.evolve_from_instance(instance, context))
    assert len(results) == len(expected)
    result_set = {(str(r[0]), tuple(sorted(r[1].status.items()))) for r in results}
    expected_set = {(str(e[0]), tuple(sorted(e[1].items()))) for e in expected}
    assert result_set == expected_set


def test_mutually_exclusive_equals_without_allowed_values():
    """Two Fn::Equals comparing the same Ref to different literals can never
    both be true, even when the parameter has no AllowedValues or is a
    pseudo-parameter (issue #4799)."""
    cfn = Template(
        None,
        {
            "Parameters": {
                "Source": {"Type": "String", "Default": "ipv4"},
            },
            "Conditions": {
                "UseIpv4": {"Fn::Equals": [{"Ref": "Source"}, "ipv4"]},
                "UseIpv6": {"Fn::Equals": [{"Ref": "Source"}, "ipv6"]},
                "UseAcct1": {"Fn::Equals": [{"Ref": "AWS::AccountId"}, "111111111111"]},
                "UseAcct2": {"Fn::Equals": [{"Ref": "AWS::AccountId"}, "222222222222"]},
            },
        },
        regions=["us-east-1"],
    )
    conditions = create_context_for_template(cfn).conditions

    # each condition is still individually reachable
    assert conditions.is_reachable("UseIpv4", True) is True
    assert conditions.is_reachable("UseIpv6", True) is True
    assert conditions.is_reachable("UseAcct1", True) is True

    # ...but the mutually exclusive pairs can never both be true
    with pytest.raises(Unsatisfiable):
        conditions.evolve({"UseIpv4": True, "UseIpv6": True})
    with pytest.raises(Unsatisfiable):
        conditions.evolve({"UseAcct1": True, "UseAcct2": True})


def test_referenced_equals_paired_once():
    """A condition's equals includes those of every condition it references,
    so the same Fn::Equals is reachable many times over. Pairing it on each
    visit makes the clause count quadratic in the references rather than in
    the distinct Fn::Equals (issue #4984)."""
    regions = 30
    cfn = Template(
        None,
        {
            "Conditions": {
                **{
                    f"IsRegion{i}": {
                        "Fn::Equals": [{"Ref": "AWS::Region"}, f"region-{i}"]
                    }
                    for i in range(regions)
                },
                **{
                    f"InGroup{i}": {
                        "Fn::Or": [
                            {"Condition": f"IsRegion{(i + j) % regions}"}
                            for j in range(10)
                        ]
                    }
                    for i in range(regions)
                },
            },
            "Resources": {},
        },
        regions=["us-east-1"],
    )
    conditions = create_context_for_template(cfn).conditions

    assert len(conditions.cnf.data) < 2 * regions * regions

    # the mutual exclusion from #4799 still holds
    assert conditions.is_reachable("IsRegion0", True) is True
    assert conditions.is_reachable("IsRegion1", True) is True
    with pytest.raises(Unsatisfiable):
        conditions.evolve({"IsRegion0": True, "IsRegion1": True})

    # a group is satisfiable, and excludes a region none of its members allow
    in_group = conditions.evolve({"InGroup0": True})
    assert in_group.is_reachable("IsRegion0", True) is True
    assert in_group.is_reachable("IsRegion20", True) is False


def test_inline_equals_mutual_exclusion():
    """An Fn::Equals reached only by traversing another condition is still
    paired, so an enclosing Fn::Or stays mutually exclusive with a sibling
    condition on the same Ref (issue #4984)."""
    cfn = Template(
        None,
        {
            "Conditions": {
                "IsEast": {"Fn::Equals": [{"Ref": "AWS::Region"}, "us-east-1"]},
                "IsElsewhere": {
                    "Fn::Or": [
                        {"Fn::Equals": [{"Ref": "AWS::Region"}, "us-west-2"]},
                        {"Fn::Equals": [{"Ref": "AWS::Region"}, "eu-west-1"]},
                    ]
                },
            },
            "Resources": {},
        },
        regions=["us-east-1"],
    )
    conditions = create_context_for_template(cfn).conditions

    assert conditions.is_reachable("IsEast", True) is True
    assert conditions.is_reachable("IsElsewhere", True) is True

    with pytest.raises(Unsatisfiable):
        conditions.evolve({"IsEast": True, "IsElsewhere": True})


def test_allowed_values_covered_through_reference():
    """The Or that forces a fully covered AllowedValues parameter is still
    emitted when the covering Fn::Equals are only reachable by traversing
    another condition. Deduping on the Ref rather than on the Fn::Equals
    would leave a value uncovered and drop the Or (issue #4984)."""
    cfn = Template(
        None,
        {
            "Parameters": {
                "Stage": {"Type": "String", "AllowedValues": ["beta", "prod"]},
            },
            "Conditions": {
                "IsBetaOrProd": {
                    "Fn::Or": [
                        {"Fn::Equals": [{"Ref": "Stage"}, "beta"]},
                        {"Fn::Equals": [{"Ref": "Stage"}, "prod"]},
                    ]
                },
            },
            "Resources": {},
        },
        regions=["us-east-1"],
    )
    conditions = create_context_for_template(cfn).conditions

    # Stage can only be beta or prod, so a condition covering both is always true
    assert conditions.is_reachable("IsBetaOrProd", True) is True
    assert conditions.is_reachable("IsBetaOrProd", False) is False


def test_condition_failures():
    with pytest.raises(ValueError):
        Conditions.create_from_instance([], {}, {})


def test_condition_non_string_names_skipped():
    # Regression test for https://github.com/aws-cloudformation/cfn-lint/issues/4813
    # A non-string condition logical ID must not crash create_from_instance
    # (sympy Symbol requires a string name).  The invalid name is skipped
    # here and reported by rule E8001.
    conditions = Conditions.create_from_instance(
        {123: {"Fn::Equals": ["a", "b"]}, "IsValid": {"Fn::Equals": ["a", "b"]}},
        {},
        {},
    )
    assert 123 not in conditions.conditions
    assert "IsValid" in conditions.conditions


def test_is_reachable_unknown_condition():
    cfn = Template(None, template(), regions=["us-east-1"])
    context = create_context_for_template(cfn)

    # An unknown condition is conservatively considered reachable.
    assert context.conditions.is_reachable("DoesNotExist", True) is True
    assert context.conditions.is_reachable("DoesNotExist", False) is True


def test_is_reachable_fact_vs_assumption():
    cfn = Template(None, template(), regions=["us-east-1"])
    base = create_context_for_template(cfn).conditions

    # A fact (e.g. a resource Condition or enclosing Fn::If) forces the value,
    # so the opposite is unreachable.
    fact = base.evolve({"IsUsEast1": True})
    assert fact.is_reachable("IsUsEast1", True) is True
    assert fact.is_reachable("IsUsEast1", False) is False

    # An assumption from scenario enumeration does not force anything, so both
    # values remain reachable.
    assumed = base.evolve({"IsUsEast1": True}, assumed=True)
    assert assumed.is_reachable("IsUsEast1", True) is True
    assert assumed.is_reachable("IsUsEast1", False) is True


def test_is_reachable_implication():
    cfn = Template(None, template(), regions=["us-east-1"])
    base = create_context_for_template(cfn).conditions

    # IsUsEast1AndProd = IsUsEast1 AND IsProd, so pinning it True as a fact
    # forces both operands True; their False values are unreachable.
    fact = base.evolve({"IsUsEast1AndProd": True})
    assert fact.is_reachable("IsUsEast1", True) is True
    assert fact.is_reachable("IsUsEast1", False) is False
    assert fact.is_reachable("IsProd", False) is False

    # As an assumption it forces nothing.
    assumed = base.evolve({"IsUsEast1AndProd": True}, assumed=True)
    assert assumed.is_reachable("IsUsEast1", False) is True


def test_or_of_and_pairs_does_not_explode():
    """An Fn::Or of Fn::And pairs is encoded with a linear number of
    clauses (issue #4737). Converting it to CNF by distribution would
    create 2^30 clauses."""
    pairs = 30
    cfn = Template(
        None,
        {
            "Parameters": {
                "Stage": {"Type": "String", "AllowedValues": ["beta", "prod"]},
            },
            "Conditions": {
                "IsDeployed": {
                    "Fn::Or": [
                        {
                            "Fn::And": [
                                {"Fn::Equals": [{"Ref": "AWS::Region"}, f"region-{i}"]},
                                {"Fn::Equals": [{"Ref": "Stage"}, "prod"]},
                            ]
                        }
                        for i in range(pairs)
                    ]
                },
                "IsProd": {"Fn::Equals": [{"Ref": "Stage"}, "prod"]},
            },
            "Resources": {},
        },
        regions=["us-east-1"],
    )
    conditions = create_context_for_template(cfn).conditions

    assert len(conditions.cnf.data) < 50 * pairs

    deployed = conditions.evolve({"IsDeployed": True})
    assert deployed.is_reachable("IsProd", True) is True
    assert deployed.is_reachable("IsProd", False) is False

    with pytest.raises(Unsatisfiable):
        conditions.evolve({"IsDeployed": True, "IsProd": False})

    not_prod = conditions.evolve({"IsProd": False})
    assert not_prod.is_reachable("IsDeployed", True) is False


def test_each_set_of_pins_is_solved_once():
    """Every Fn::If and condition scenario asks whether its pins are
    satisfiable, and most ask about the same pins. Solving each set once
    keeps the solver calls from growing with the number of Fn::If
    (issue #4991)."""
    cfn = Template(None, template(), regions=["us-east-1"])
    base = create_context_for_template(cfn).conditions

    with patch("cfnlint.conditions._solver.satisfiable", wraps=satisfiable) as solver:
        first = base.evolve({"IsUsEast1": True})
        # The first question also checks, once, that the template's clauses
        # are satisfiable on their own (one solve per connected component).
        calls = solver.call_count
        base.evolve({"IsUsEast1": True})
        base.evolve({"IsUsEast1": True}, assumed=True)
        assert solver.call_count == calls

        # The same pins reached in a different order or from another branch.
        first.evolve({"IsProd": True})
        base.evolve({"IsProd": True}).evolve({"IsUsEast1": True})
        base.evolve({"IsProd": True, "IsUsEast1": True})
        assert solver.call_count == calls + 2

        # A reachability question is a set of pins too.
        assert first.is_reachable("IsProd", True) is True
        assert first.evolve({"IsProd": True}).is_reachable("IsUsEast1", True)
        assert solver.call_count == calls + 2

        # An unsatisfiable set is remembered as well.
        for _ in range(2):
            with pytest.raises(Unsatisfiable):
                first.evolve({"IsUsWest2": True})
        assert solver.call_count == calls + 3


def test_pins_keep_facts_and_assumptions_apart():
    cfn = Template(None, template(), regions=["us-east-1"])
    base = create_context_for_template(cfn).conditions

    # Same statuses after the second pin, but only the fact forces IsUsEast1.
    assumed = base.evolve({"IsUsEast1": True}, assumed=True).evolve({"IsProd": True})
    fact = base.evolve({"IsUsEast1": True}).evolve({"IsProd": True})
    assert assumed.status == fact.status
    assert assumed.is_reachable("IsUsEast1", False) is True
    assert fact.is_reachable("IsUsEast1", False) is False


def test_cnf_is_the_template_clauses_plus_one_per_pin():
    cfn = Template(None, template(), regions=["us-east-1"])
    base = create_context_for_template(cfn).conditions
    clauses = len(base.cnf.data)

    pinned = base.evolve({"IsUsEast1": True}, assumed=True).evolve({"IsProd": True})
    assert len(pinned.cnf.data) == clauses + 2
    assert len(pinned.hard_cnf.data) == clauses + 1

    # Each access is a new copy, so changing one does not reach later answers:
    # this evolve would raise Unsatisfiable if Not(IsProd) had been kept.
    pinned.cnf.add_prop(Not(Symbol("IsProd")))
    assert len(pinned.cnf.data) == clauses + 2
    pinned.evolve({"IsAi": True})


def _alarms_reusing_conditions(alarms):
    return {
        "Parameters": {"Stage": {"Type": "String", "AllowedValues": ["beta", "prod"]}},
        "Conditions": {
            "IsProd": {"Fn::Equals": [{"Ref": "Stage"}, "prod"]},
            "IsEast": {"Fn::Equals": [{"Ref": "AWS::Region"}, "us-east-1"]},
            "IsProdEast": {
                "Fn::And": [{"Condition": "IsProd"}, {"Condition": "IsEast"}]
            },
        },
        "Resources": {
            f"Alarm{i}": {
                "Type": "AWS::CloudWatch::Alarm",
                # Fn::If inside a conditional resource evolves the resource's
                # Conditions, so the cache must reach evolved objects too.
                "Condition": "IsProd",
                "Properties": {
                    "Namespace": "Service",
                    "MetricName": f"Metric{i}",
                    "Statistic": "Sum",
                    "ComparisonOperator": "GreaterThanThreshold",
                    "EvaluationPeriods": 1,
                    "Threshold": {"Fn::If": ["IsProdEast", 1, 10]},
                    "Period": {"Fn::If": ["IsEast", 300, 60]},
                },
            }
            for i in range(alarms)
        },
    }


def test_solver_calls_do_not_grow_with_reused_conditions():
    """Resources that reuse the same conditions in Fn::If ask the same
    questions, so a whole lint solves the same number of pin sets however
    many such resources there are (issue #4991)."""
    calls = []
    for alarms in (5, 20):
        with patch(
            "cfnlint.conditions._solver.satisfiable", wraps=satisfiable
        ) as solver:
            matches = lint(
                json.dumps(_alarms_reusing_conditions(alarms)),
                config=ManualArgs(regions=["us-east-1"]),
            )
        assert matches == []
        calls.append(solver.call_count)

    assert calls[0] > 0
    assert calls[0] == calls[1]


def test_scenarios_do_not_depend_on_name_order():
    """Past the cap, the scenarios evolve_from_instance enumerates depend only
    on the condition names, not on set iteration order (hash seed)."""
    names = [f"C{i}" for i in range(10)]
    cfn = Template(
        "",
        {
            "Parameters": {f"P{i}": {"Type": "String"} for i in range(10)},
            "Conditions": {
                f"C{i}": {"Fn::Equals": [{"Ref": f"P{i}"}, "x"]} for i in range(10)
            },
            "Resources": {},
        },
    )
    context = create_context_for_template(cfn)

    def scenarios(order):
        class _Ordered(set):
            def __iter__(self):
                return iter(order)

        with patch(
            "cfnlint.context.conditions._conditions.get_conditions_from_property",
            return_value=_Ordered(order),
        ):
            return [
                sorted(s.status.items())
                for _, s in context.conditions.evolve_from_instance({}, context)
            ]

    forward = scenarios(names)
    assert len(forward) == 128
    assert forward == scenarios(names[::-1])


def test_truncated_scenarios_keep_both_ends_of_the_order():
    """Past the cap, the scenarios are taken alternately from the two ends
    of the True-first order over the sorted condition names, so a nested
    Fn::If's first branch (its condition true) and its last leaf (every
    condition false) are both resolved whatever the solver."""
    names = [f"C{i}" for i in range(10)]
    cfn = Template(
        "",
        {
            "Parameters": {f"P{i}": {"Type": "String"} for i in range(10)},
            "Conditions": {
                f"C{i}": {"Fn::Equals": [{"Ref": f"P{i}"}, "x"]} for i in range(10)
            },
            "Resources": {},
        },
    )
    context = create_context_for_template(cfn)
    with patch(
        "cfnlint.context.conditions._conditions.get_conditions_from_property",
        return_value=set(names),
    ):
        scenarios = {
            tuple(s.status[n] for n in sorted(names))
            for _, s in context.conditions.evolve_from_instance({}, context)
        }
    assert len(scenarios) == 128
    everything = sorted(
        itertools.product((True, False), repeat=10),
        key=lambda m: tuple(not v for v in m),
    )
    assert scenarios == set(everything[:64]) | set(everything[-64:])


@pytest.mark.parametrize("reverse", [False, True])
def test_self_equals_does_not_exclude_other_equals(reverse):
    """Two buckets with one name, under P == "x" and under P == P: both exist
    when P is "x", so E3019 reports the clash whatever the definition order."""
    definitions = [
        ("DynamicFirst", {"Fn::Equals": [{"Ref": "P"}, "x"]}),
        ("StaticSecond", {"Fn::Equals": [{"Ref": "P"}, {"Ref": "P"}]}),
    ]
    if reverse:
        definitions.reverse()
    template = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Parameters": {"P": {"Type": "String"}},
        "Conditions": dict(definitions),
        "Resources": {
            name: {
                "Type": "AWS::S3::Bucket",
                "Condition": condition,
                "Properties": {"BucketName": "same-name"},
            }
            for name, condition in (
                ("First", "DynamicFirst"),
                ("Second", "StaticSecond"),
            )
        },
    }
    matches = lint(json.dumps(template), config=ManualArgs(regions=["us-east-1"]))
    assert sorted(m.rule.id for m in matches) == ["E3019", "E3019", "W8003"]


@pytest.mark.parametrize(
    "equals,always",
    [
        (["Prod", "Prod"], True),
        (["Prod", "Dev"], False),
        ([{"Ref": "P"}, {"Ref": "P"}], True),
    ],
)
def test_static_equals_is_a_constant(equals, always):
    """A static Fn::Equals has one value, as in the template-level engine."""
    cfn = Template(
        "",
        {
            "Parameters": {"P": {"Type": "String"}},
            "Conditions": {"Static": {"Fn::Equals": equals}},
            "Resources": {},
        },
    )
    conditions = create_context_for_template(cfn).conditions
    assert conditions.is_reachable("Static", always)
    assert not conditions.is_reachable("Static", not always)
    assert cfn.conditions.is_always_true("Static") is always


def test_rules_restrict_conditions():
    """A Rules assertion restricts parameter values for context checks, as it
    does in the template-level engine."""
    cfn = Template(
        "",
        {
            "Parameters": {
                "Env": {"Type": "String", "AllowedValues": ["dev", "prod"]},
                "Region": {"Type": "String"},
            },
            "Rules": {
                "ProdOnly": {"Assertions": [{"Assert": {"Condition": "IsProd"}}]},
                "EastWhenProd": {
                    "RuleCondition": {"Fn::Equals": [{"Ref": "Env"}, "prod"]},
                    "Assertions": [
                        {"Assert": {"Fn::Equals": [{"Ref": "Region"}, "east"]}}
                    ],
                },
                "Ignored": {
                    "Assertions": [
                        {"Assert": {"Fn::Contains": [["a"], {"Ref": "Env"}]}}
                    ]
                },
            },
            "Conditions": {
                "IsProd": {"Fn::Equals": [{"Ref": "Env"}, "prod"]},
                "IsDev": {"Fn::Equals": [{"Ref": "Env"}, "dev"]},
                "IsWest": {"Fn::Equals": [{"Ref": "Region"}, "west"]},
            },
            "Resources": {},
        },
    )
    conditions = create_context_for_template(cfn).conditions
    # the template's Rules are read, not reordered
    assert cfn.template["Rules"]["EastWhenProd"]["RuleCondition"] == {
        "Fn::Equals": [{"Ref": "Env"}, "prod"]
    }
    assert not conditions.is_reachable("IsDev", True)
    assert not conditions.is_reachable("IsProd", False)
    assert not conditions.is_reachable("IsWest", True)
    assert conditions.is_reachable("IsWest", False)
    # the template-level engine gives the same answers
    assert cfn.conditions.is_always_true("IsProd")
    assert list(cfn.conditions.build_scenarios({"IsWest": None})) == [{"IsWest": False}]


@pytest.mark.parametrize(
    "assertions",
    [
        # Env would have to be prod and dev at once
        [
            {"Fn::Equals": [{"Ref": "Env"}, "prod"]},
            {"Fn::Equals": [{"Ref": "Env"}, "dev"]},
        ],
        [{"Fn::Equals": ["x", "y"]}],  # false whatever the parameters
    ],
)
def test_rules_that_can_never_hold_are_left_out(assertions):
    """When no parameter values pass the Rules, the context engine answers
    about the conditions alone, as for a template without Rules: otherwise
    every value of every condition is unreachable, including conditions on
    parameters the Rules never mention.  The template-level engine keeps
    answering that nothing is satisfiable, which is what stops it resolving
    parameter values on such a template."""
    template = {
        "Parameters": {
            "Env": {"Type": "String", "AllowedValues": ["prod", "dev"]},
            "Size": {"Type": "String"},
        },
        "Rules": {"Never": {"Assertions": [{"Assert": a} for a in assertions]}},
        "Conditions": {
            "IsProd": {"Fn::Equals": [{"Ref": "Env"}, "prod"]},
            "IsBig": {"Fn::Equals": [{"Ref": "Size"}, "big"]},
        },
        "Resources": {},
    }
    cfn = Template("", template)
    conditions = create_context_for_template(cfn).conditions
    for name in ("IsProd", "IsBig"):
        for value in (True, False):
            assert conditions.is_reachable(name, value)
    # the same template without the Rules section gives the same clauses
    del template["Rules"]
    same = create_context_for_template(Template("", template)).conditions
    assert same.cnf.data == conditions.cnf.data
    # the template-level engine answers as it does without the fix
    assert not cfn.conditions.satisfiable({}, {})
    assert not cfn.conditions.satisfiable({"IsBig": True}, {"Size": "big"})


def test_evolve_returns_one_conditions_per_set_of_pins():
    """Evolving to the same pins, from anywhere, gives the same object, and
    each condition is evolved to a status once per template."""
    cfn = Template("", template())
    conditions = create_context_for_template(cfn).conditions
    first = conditions.evolve({"IsUsEast1": True}).evolve({"IsProd": False})
    second = conditions.evolve({"IsProd": False}).evolve({"IsUsEast1": True})
    assert first is second
    assert first.status == {"IsUsEast1": True, "IsProd": False}
    # an assumed pin is not a fact, so it is a different Conditions
    assumed = conditions.evolve({"IsUsEast1": True}, assumed=True)
    assert assumed is not conditions.evolve({"IsUsEast1": True})
    assert assumed.is_reachable("IsUsEast1", False)


def test_condition_evolve_calls_do_not_grow_with_resources():
    """Resources that reuse the same conditions evolve each condition to each
    status once per lint, however many such resources there are."""
    calls = []
    for alarms in (5, 20):
        with patch.object(
            Condition, "evolve", autospec=True, side_effect=Condition.evolve
        ) as evolve:
            lint(
                json.dumps(_alarms_reusing_conditions(alarms)),
                config=ManualArgs(regions=["us-east-1"]),
            )
        calls.append(evolve.call_count)

    assert calls[0] > 0
    assert calls[0] == calls[1]


def test_true_and_one_are_different_equals():
    """Fn::Equals [Ref P, true] and Fn::Equals [Ref P, 1] compare P with the
    strings "true" and "1", so they are two Fn::Equals with two solver
    variables (True == 1 in Python, so the hashes must tell them apart)."""
    conditions = Conditions.create_from_instance(
        {
            "C0": {"Fn::Equals": [{"Ref": "P0"}, True]},
            "T": {"Fn::Not": [{"Fn::Equals": [{"Ref": "P0"}, 1]}]},
        },
        {},
        {},
    )
    hashes = {
        eql.hash
        for condition in conditions.conditions.values()
        for eql in condition.equals
    }
    assert len(hashes) == 2
    # P0 = "true" satisfies both
    evolved = conditions.evolve({"C0": True, "T": True})
    assert evolved.status == {"C0": True, "T": True}
    # P0 = "1" satisfies neither; the two equals are still exclusive
    with pytest.raises(Unsatisfiable):
        conditions.evolve({"C0": True, "T": False})


def test_literals_compare_as_cloudformation_strings():
    """CloudFormation compares Fn::Equals operands as strings and writes a
    boolean as true/false: true equals "true" but not "True" or 1, 1 equals
    "1" but not 1.0, false equals neither 0 nor "False"."""
    cases = {
        "BoolTrue": ([True, "true"], True),
        "BoolCase": ([True, "True"], False),
        "BoolOne": ([True, 1], False),
        "FalseZero": ([False, 0], False),
        "FalseStr": (["false", False], True),
        "IntStr": (["1", 1], True),
        "IntFloat": ([1, 1.0], False),
    }
    conditions = Conditions.create_from_instance(
        {k: {"Fn::Equals": v} for k, (v, _) in cases.items()}, {}, {}
    )
    for name, (_, expected) in cases.items():
        (equal,) = conditions.conditions[name].equals
        assert equal.is_static is expected, name


def test_bool_and_string_literal_are_one_equals():
    """Fn::Equals [Ref P, true] and [Ref P, "true"] are one comparison, so
    pinning one pins the other; the template keeps its boolean."""
    instance = {
        "A": {"Fn::Equals": [{"Ref": "P"}, True]},
        "B": {"Fn::Equals": ["true", {"Ref": "P"}]},
        "C": {"Fn::Equals": [{"Ref": "P"}, 1]},
    }
    conditions = Conditions.create_from_instance(instance, {}, {})
    (a,) = conditions.conditions["A"].equals
    (b,) = conditions.conditions["B"].equals
    (c,) = conditions.conditions["C"].equals
    assert a.hash == b.hash
    assert a.hash != c.hash
    assert True in instance["A"]["Fn::Equals"]
    with pytest.raises(Unsatisfiable):
        conditions.evolve({"A": True, "B": False})
    # P = "true" makes A and B true and C false
    assert conditions.evolve({"A": True, "B": True}).status == {"A": True, "B": True}
    with pytest.raises(Unsatisfiable):
        conditions.evolve({"A": True, "C": True})


def test_allowed_values_compare_as_cloudformation_strings():
    """AllowedValues [true, false] are "true" and "false", and [1, 2] are
    "1" and "2", so the equals over each list cannot all be false."""
    cfn = Template(
        None,
        {
            "Parameters": {
                "P": {"Type": "String", "AllowedValues": [True, False]},
                "N": {"Type": "Number", "AllowedValues": [1, 2]},
            },
            "Conditions": {
                "T": {"Fn::Equals": [{"Ref": "P"}, "true"]},
                "F": {"Fn::Equals": [{"Ref": "P"}, False]},
                "One": {"Fn::Equals": [{"Ref": "N"}, "1"]},
                "Two": {"Fn::Equals": [{"Ref": "N"}, 2]},
            },
        },
        regions=["us-east-1"],
    )
    conditions = create_context_for_template(cfn).conditions
    with pytest.raises(Unsatisfiable):
        conditions.evolve({"T": False, "F": False})
    with pytest.raises(Unsatisfiable):
        conditions.evolve({"One": False, "Two": False})
    with pytest.raises(Unsatisfiable):
        conditions.evolve({"T": True, "F": True})
    conditions.evolve({"T": True, "F": False})


def _queues_with_shared_conditions(n, private=2, shared=6):
    """n SQS queues, each with 8 properties set by Fn::If on 8 conditions: 2
    private to the queue and 6 shared, so the lint asks up to 128 scenarios
    per queue over 2n + 6 conditions."""
    props = [
        ("DelaySeconds", 0, 5),
        ("MaximumMessageSize", 1024, 2048),
        ("MessageRetentionPeriod", 60, 120),
        ("ReceiveMessageWaitTimeSeconds", 0, 10),
        ("VisibilityTimeout", 30, 60),
        ("KmsDataKeyReusePeriodSeconds", 300, 600),
        ("SqsManagedSseEnabled", True, False),
        ("QueueName", None, None),
    ]
    params = {f"S{j}": {"Type": "String"} for j in range(shared)}
    conds = {f"S{j}": {"Fn::Equals": [{"Ref": f"S{j}"}, "a"]} for j in range(shared)}
    res = {}
    for i in range(n):
        p = {}
        for j, (name, a, b) in enumerate(props):
            if j < private:
                params[f"P{i}x{j}"] = {"Type": "String"}
                conds[f"C{i}x{j}"] = {"Fn::Equals": [{"Ref": f"P{i}x{j}"}, "a"]}
                c = f"C{i}x{j}"
            else:
                c = f"S{j - private}"
            p[name] = (
                {"Fn::If": [c, f"q{i}-a", f"q{i}-b"]}
                if a is None
                else {"Fn::If": [c, a, b]}
            )
        res[f"Queue{i}"] = {"Type": "AWS::SQS::Queue", "Properties": p}
    return {"Parameters": params, "Conditions": conds, "Resources": res}


def test_evolved_conditions_hold_only_their_pins():
    """An evolved Conditions lays its pins over the template's condition dict
    instead of copying every condition (one full dict per
    scenario reached 2 GB on 400 queues)."""
    cfn = Template("", _queues_with_shared_conditions(50))
    root = create_context_for_template(cfn).conditions
    assert isinstance(root.conditions, dict)
    names = list(root.conditions)
    assert len(names) == 106

    first = root.evolve({"S0": True, "C3x1": False})
    second = first.evolve({"C7x0": True}, assumed=True)
    for evolved, pinned in ((first, 2), (second, 3)):
        assert isinstance(evolved.conditions, _Pinned)
        assert evolved.conditions.base is root.conditions
        assert len(evolved.conditions.pinned) == pinned
        # the mapping still reads as the whole template
        assert list(evolved.conditions) == names
        assert len(evolved.conditions) == 106
        assert "S5" in evolved.conditions and evolved.conditions["S5"].status is None
        assert evolved.conditions.get("Nope") is None
    assert first.conditions["S0"].status is True
    assert first.conditions["C3x1"].status is False
    assert second.conditions["C7x0"].status is True
    # statuses come back in template order, whatever the pin order
    assert list(second.status) == ["S0", "C3x1", "C7x0"]
    assert second.status == {"S0": True, "C3x1": False, "C7x0": True}
    assert second._hard_pins == frozenset({("S0", True), ("C3x1", False)})
    assert second._pins == frozenset({("S0", True), ("C3x1", False), ("C7x0", True)})
    # the same pins give the same object whatever the order of the pins
    assert root.evolve({"C3x1": False}).evolve({"S0": True}) is first
    with pytest.raises(Unsatisfiable):
        second.evolve({"S0": False})
    # a mapping that equals the dict it would have been
    assert dict(second.conditions) == {
        **root.conditions,
        "S0": first.conditions["S0"],
        "C3x1": first.conditions["C3x1"],
        "C7x0": second.conditions["C7x0"],
    }


class _NoIteration(dict):
    """A condition dict that refuses to be iterated."""

    def __iter__(self):
        raise AssertionError("iterated over every condition")

    def items(self):
        raise AssertionError("iterated over every condition")

    def values(self):
        raise AssertionError("iterated over every condition")


def test_evolve_and_status_read_only_the_pins():
    """Once the template's own facts are known, evolving a Conditions and
    reading its status touch the pins laid over the template, never every
    condition of the template."""
    cfn = Template("", _queues_with_shared_conditions(50))
    root = create_context_for_template(cfn).conditions
    first = root.evolve({"S0": True})
    first.conditions.base = _NoIteration(root.conditions)
    second = first.evolve({"C3x1": False, "S1": True}, assumed=True)
    assert first.status == {"S0": True}
    assert list(second.status) == ["S0", "S1", "C3x1"]
    assert second.evolve({"S1": True}) is second
    assert second.is_reachable("S2", True)
    with pytest.raises(Unsatisfiable):
        second.evolve({"S0": False})


def test_evolved_memo_is_bounded():
    """The memo of evolved Conditions is emptied past _EVOLVED_MAX entries
    and evolving keeps working."""
    cfn = Template("", _queues_with_shared_conditions(4))
    root = create_context_for_template(cfn).conditions
    with patch("cfnlint.context.conditions._conditions._EVOLVED_MAX", 8):
        kept = []
        for i in range(4):
            for j in range(2):
                for value in (True, False):
                    evolved = root.evolve({f"C{i}x{j}": value})
                    assert evolved.status == {f"C{i}x{j}": value}
                    kept.append(evolved)
        assert len({id(c) for c in kept}) == 16
        assert len(root._evolved) <= 8


def test_lint_memory_does_not_grow_with_scenarios():
    """Linting many queues keeps one small dict per scenario: the evolved
    memo holds no copy of the template's conditions."""
    cfn = Template("", _queues_with_shared_conditions(30))
    root = create_context_for_template(cfn).conditions
    for name, resource in cfn.template["Resources"].items():
        for _ in root.evolve_from_instance(
            resource["Properties"], create_context_for_template(cfn)
        ):
            pass
    assert len(root._evolved) > 1000
    assert max(len(c.conditions.pinned) for c in root._evolved.values()) <= 8
    assert all(c.conditions.base is root.conditions for c in root._evolved.values())


def _condition_chain(n, top_down):
    """C0 is an Fn::Equals and each Ci negates C(i-1); written from the top
    of the chain down when ``top_down``, so that the first condition read
    references every other one."""
    conditions = {"C0": {"Fn::Equals": [{"Ref": "P"}, "a"]}}
    for i in range(1, n):
        conditions[f"C{i}"] = {"Fn::Not": [{"Condition": f"C{i - 1}"}]}
    if top_down:
        conditions = dict(reversed(conditions.items()))
    return {
        "Parameters": {"P": {"Type": "String"}},
        "Conditions": conditions,
        "Resources": {},
    }


@pytest.mark.parametrize("top_down", [False, True])
def test_long_condition_chain_in_either_order(top_down):
    """A Condition: chain of 400 links is built and answered whether the
    template lists it bottom-up or top-down (the walks over equals and
    is_region are settled link by link, so no walk recurses along the chain)."""
    n = 400
    cfn = Template("", _condition_chain(n, top_down))
    conditions = create_context_for_template(cfn).conditions
    assert len(conditions.conditions) == n
    last = f"C{n - 1}"
    assert conditions.conditions[last].equals[0].hash == (
        conditions.conditions["C0"].equals[0].hash
    )
    assert not conditions.conditions[last].is_region
    # C(n-1) is C0 negated n-1 times, so pinning it fixes C0
    pinned = conditions.evolve({last: True})
    assert pinned.is_reachable("C0", n % 2 == 1)
    assert not pinned.is_reachable("C0", n % 2 == 0)
