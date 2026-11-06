"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

import string
from unittest import TestCase

from cfnlint.conditions import UnknownSatisfisfaction
from cfnlint.conditions._utils import get_hash
from cfnlint.decode import decode_str
from cfnlint.template import Template


class TestConditions(TestCase):
    """Test Conditions"""

    def test_bad_condition_definition(self):
        """Badly formatted condition statements will return no results"""
        template = decode_str(
            """
        Conditions:
          IsProd: !Equals [!Ref Environment, "prod", "production"]
          IsUsEast1: !Equals [!Ref "AWS::Region", "us-east-1"]
        """
        )[0]

        cfn = Template("", template)
        self.assertEqual(
            len(cfn.conditions._conditions), 1
        )  # would be 2 but IsProd fails
        # test coverage for KeyErrors in the following functions
        self.assertTrue(
            cfn.conditions.check_implies(
                {
                    "Test": True,
                },
                "IsUsEast1",
            )
        )
        self.assertEqual(
            list(cfn.conditions.build_scenarios({"IsProd": None, "IsUsEast1": None})),
            [],
        )

    def test_run_away_scenarios(self):
        """We cap runaway scenarios"""
        template = {
            "Parameters": {},
            "Conditions": {},
        }
        condition_names = {}
        for p in string.ascii_letters[0:10]:
            template["Parameters"][f"{p}Parameter"] = {
                "Type": "String",
            }
            template["Conditions"][f"{p}Condition"] = {
                "Fn::Equals": [{"Ref": f"{p}Parameter"}, "{p}"]
            }
            condition_names[f"{p}Condition"] = None

        cfn = Template("", template)
        self.assertEqual(len(cfn.conditions._conditions), 10)
        self.assertEqual(
            len(list(cfn.conditions.build_scenarios(condition_names))),
            cfn.conditions._max_scenarios,
        )

    def test_scenarios_do_not_depend_on_name_order(self):
        """Past the cap, which scenarios are tried depends only on the
        condition names, not on the order a caller collected them in"""
        template = {"Parameters": {}, "Conditions": {}}
        for i in range(10):
            template["Parameters"][f"P{i}"] = {"Type": "String"}
            template["Conditions"][f"C{i}"] = {"Fn::Equals": [{"Ref": f"P{i}"}, "x"]}
        cfn = Template("", template)
        names = [f"C{i}" for i in range(10)]
        forward = list(cfn.conditions.build_scenarios(dict.fromkeys(names)))
        backward = list(cfn.conditions.build_scenarios(dict.fromkeys(names[::-1])))
        self.assertEqual(len(forward), cfn.conditions._max_scenarios)
        # assertTrue: a failing assertEqual would diff 128 long lists
        self.assertTrue(
            [sorted(s.items()) for s in forward]
            == [sorted(s.items()) for s in backward]
        )

    def test_self_equals_constrains_nothing(self):
        """Ref P against Ref P is always true and says nothing about P, so it
        does not exclude another Fn::Equals on P, in either definition order"""
        for names in (
            ("DynamicFirst", "StaticSecond"),
            ("StaticSecond", "DynamicFirst"),
        ):
            definitions = {
                "DynamicFirst": {"Fn::Equals": [{"Ref": "P"}, "x"]},
                "StaticSecond": {"Fn::Equals": [{"Ref": "P"}, {"Ref": "P"}]},
            }
            cfn = Template(
                "",
                {
                    "Parameters": {"P": {"Type": "String"}},
                    "Conditions": {name: definitions[name] for name in names},
                },
            )
            self.assertTrue(
                cfn.conditions.satisfiable({"DynamicFirst": True}, {"P": "x"})
            )
            self.assertTrue(cfn.conditions.is_always_true("StaticSecond"))
            self.assertEqual(
                list(
                    cfn.conditions.build_scenarios(
                        {"DynamicFirst": None, "StaticSecond": None}
                    )
                ),
                [
                    {"DynamicFirst": True, "StaticSecond": True},
                    {"DynamicFirst": False, "StaticSecond": True},
                ],
            )

    def test_check_implies(self):
        """We properly validate implies scenarios"""
        template = {
            "Parameters": {},
            "Conditions": {},
        }
        condition_names = []
        for p in string.ascii_letters[0:4]:
            template["Parameters"][f"{p}Parameter"] = {
                "Type": "String",
            }
            template["Conditions"][f"{p}Condition"] = {
                "Fn::Equals": [{"Ref": f"{p}Parameter"}, "{p}"]
            }
            condition_names.append(f"{p}Condition")

        cfn = Template("", template)
        self.assertFalse(
            cfn.conditions.check_implies({"aCondition": False}, "aCondition")
        )
        self.assertTrue(
            cfn.conditions.check_implies({"aCondition": True}, "aCondition")
        )
        self.assertTrue(
            cfn.conditions.check_implies(
                {"aCondition": True, "bCondition": False}, "aCondition"
            )
        )

    def test_check_always_true_or_false(self):
        """We properly validate static equals"""
        template = decode_str(
            """
        Parameters:
          FalseParameter:
            Default: "false"
            Type: String
        Conditions:
          IsTrue: !Equals ["true", "true"]
          IsFalse: !Equals [!Ref FalseParameter, !Ref FalseParameter]
        """
        )[0]

        cfn = Template("", template)
        self.assertEqual(len(cfn.conditions._conditions), 2)
        # test coverage for KeyErrors in the following functions
        self.assertTrue(cfn.conditions.check_implies({"IsTrue": True}, "IsFalse"))

    def test_check_never_false(self):
        """With allowed values two conditions can not both be false"""
        template = decode_str(
            """
        Parameters:
          Environment:
            Type: String
            AllowedValues: ["prod", "dev"]
        Conditions:
          IsProd: !Equals [!Ref Environment, "prod"]
          IsDev: !Equals [!Ref Environment, "dev"]
        """
        )[0]

        cfn = Template("", template)
        self.assertEqual(len(cfn.conditions._conditions), 2)
        self.assertListEqual(
            list(cfn.conditions.build_scenarios({"IsProd": None, "IsDev": None})),
            # scenarios follow the sorted condition names
            [
                {"IsDev": True, "IsProd": False},
                {"IsDev": False, "IsProd": True},
            ],
        )

    def test_check_can_be_false(self):
        """With allowed values two conditions can both be false"""
        template = decode_str(
            """
        Parameters:
          Environment:
            Type: String
            AllowedValues: ["prod", "dev", "stage"]
        Conditions:
          IsProd: !Equals [!Ref Environment, "prod"]
          IsDev: !Equals [!Ref Environment, "dev"]
        """
        )[0]

        cfn = Template("", template)
        self.assertEqual(len(cfn.conditions._conditions), 2)
        self.assertListEqual(
            list(cfn.conditions.build_scenarios({"IsProd": None, "IsDev": None})),
            # scenarios follow the sorted condition names
            [
                {"IsDev": True, "IsProd": False},
                {"IsDev": False, "IsProd": True},
                {"IsProd": False, "IsDev": False},
            ],
        )
        self.assertListEqual(
            list(cfn.conditions.build_scenarios({"IsProd": {True}, "IsDev": None})),
            [
                {"IsProd": True, "IsDev": False},
            ],
        )
        self.assertListEqual(
            list(cfn.conditions.build_scenarios({"IsProd": {False}, "IsDev": None})),
            [
                {"IsProd": False, "IsDev": True},
                {"IsProd": False, "IsDev": False},
            ],
        )

    def test_check_can_be_good_when_condition_value(self):
        """Some times a condition Equals doesn't match to allowed values"""
        template = decode_str(
            """
        Parameters:
          Environment:
            Type: String
            AllowedValues: ["prod", "dev", "stage"]
        Conditions:
          IsGamma: !Equals [!Ref Environment, "gamma"]
          IsBeta: !Equals ["beta", !Ref Environment]
        """
        )[0]

        cfn = Template("", template)
        self.assertEqual(len(cfn.conditions._conditions), 2)
        self.assertListEqual(
            list(cfn.conditions.build_scenarios({"IsGamma": None, "IsBeta": None})),
            [
                {"IsBeta": False, "IsGamma": False},
            ],
        )
        self.assertListEqual(
            list(cfn.conditions.build_scenarios({"IsGamma": None})),
            [
                {"IsGamma": False},
            ],
        )

    def test_check_condition_region(self):
        """Regional based condition testing"""
        template = decode_str(
            """
        Parameters:
          Environment:
            Type: String
            AllowedValues: ["prod", "dev", "stage"]
        Conditions:
          IsUsEast1: !Equals [!Ref AWS::Region, "us-east-1"]
          IsUsWest2: !Equals ["us-west-2", !Ref AWS::Region]
          IsProd: !Equals [!Ref Environment, "prod"]
        """
        )[0]

        cfn = Template("", template)
        self.assertEqual(len(cfn.conditions._conditions), 3)
        self.assertListEqual(
            cfn.conditions.build_scenerios_on_region("IsUsEast1", "us-east-1"),
            [
                True,
            ],
        )
        self.assertListEqual(
            cfn.conditions.build_scenerios_on_region("IsUsEast1", "us-west-2"),
            [
                False,
            ],
        )
        self.assertListEqual(
            cfn.conditions.build_scenerios_on_region("IsUsWest2", "us-west-2"),
            [
                True,
            ],
        )
        self.assertListEqual(
            cfn.conditions.build_scenerios_on_region("IsUsWest2", "us-east-1"),
            [
                False,
            ],
        )
        self.assertListEqual(
            cfn.conditions.build_scenerios_on_region("IsProd", "us-east-1"),
            [
                True,
                False,
            ],
        )
        self.assertListEqual(
            cfn.conditions.build_scenerios_on_region("Foo", "us-east-1"),
            [
                True,
                False,
            ],
        )

    def test_test_condition(self):
        """Get condition and test"""
        template = decode_str(
            """
        Parameters:
          Environment:
            Type: String
            AllowedValues: ["prod", "dev", "stage"]
        Conditions:
          IsUsEast1: !Equals [!Ref AWS::Region, "us-east-1"]
          IsUsWest2: !Equals ["us-west-2", !Ref AWS::Region]
          IsProd: !Equals [!Ref Environment, "prod"]
          IsUsEast1AndProd: !And [!Condition IsUsEast1, !Condition IsProd]
        """
        )[0]

        h_region = get_hash({"Ref": "AWS::Region"})
        h_environment = get_hash({"Ref": "Environment"})

        cfn = Template("", template)
        self.assertTrue(cfn.conditions.get("IsUsEast1").test({h_region: "us-east-1"}))
        self.assertFalse(cfn.conditions.get("IsProd").test({h_environment: "dev"}))
        self.assertTrue(
            cfn.conditions.get("IsUsEast1AndProd").test(
                {h_region: "us-east-1", h_environment: "prod"}
            )
        )
        self.assertFalse(
            cfn.conditions.get("IsUsEast1AndProd").test(
                {h_region: "us-east-1", h_environment: "dev"}
            )
        )

    def test_check_condition_partition(self):
        """Partition based condition testing"""
        template = decode_str(
            """
        Parameters:
          Environment:
            Type: String
            AllowedValues: ["prod", "dev", "stage"]
        Conditions:
          IsIso: !Equals [!Ref AWS::Partition, "aws-iso"]
          IsChina: !Equals ["aws-cn", !Ref AWS::Partition]
          IsNotIso: !Not [!Condition IsIso]
          IsIsoEast1: !And
            - !Condition IsIso
            - !Equals [!Ref AWS::Region, "us-iso-east-1"]
          IsGovOrIso: !Or
            - !Condition IsIso
            - !Equals [!Ref AWS::Partition, "aws-us-gov"]
          IsProd: !Equals [!Ref Environment, "prod"]
          IsProdIso: !And [!Condition IsProd, !Condition IsIso]
        """
        )[0]

        cfn = Template("", template)

        def scenarios(name, region):
            return cfn.conditions.build_scenerios_on_region(name, region)

        self.assertListEqual(scenarios("IsIso", "us-iso-east-1"), [True])
        self.assertListEqual(scenarios("IsIso", "us-iso-west-1"), [True])
        self.assertListEqual(scenarios("IsIso", "us-isob-east-1"), [False])
        self.assertListEqual(scenarios("IsIso", "us-east-1"), [False])
        self.assertListEqual(scenarios("IsChina", "cn-north-1"), [True])
        self.assertListEqual(scenarios("IsChina", "us-gov-west-1"), [False])
        self.assertListEqual(scenarios("IsNotIso", "us-iso-east-1"), [False])
        self.assertListEqual(scenarios("IsNotIso", "eu-west-1"), [True])
        self.assertListEqual(scenarios("IsIsoEast1", "us-iso-east-1"), [True])
        self.assertListEqual(scenarios("IsIsoEast1", "us-iso-west-1"), [False])
        self.assertListEqual(scenarios("IsGovOrIso", "us-gov-east-1"), [True])
        self.assertListEqual(scenarios("IsGovOrIso", "cn-north-1"), [False])
        self.assertListEqual(scenarios("IsProd", "us-iso-east-1"), [True, False])
        self.assertListEqual(scenarios("IsProdIso", "us-iso-east-1"), [True, False])
        self.assertListEqual(scenarios("IsProdIso", "us-east-1"), [False])

    def test_build_scenerios_on_region_with_condition_dne(self):
        """Get condition and test"""
        template = decode_str(
            """
        Conditions:
          IsUsEast1: !Equals [!Ref AWS::Region, "us-east-1"]
        """
        )[0]

        cfn = Template("", template)
        self.assertListEqual(
            list(cfn.conditions.build_scenerios_on_region("IsProd", "us-east-1")),
            [True, False],
        )

    def test_satifaction(self):
        """Get condition and test"""
        template = decode_str(
            """
        Parameters:
          SecurityGroups:
            Default: ""
            Type: CommaDelimitedList
        Conditions:
          IsUsEast1: !Equals [!Ref AWS::Region, "us-east-1"]
          HasSecurityGroups: !Not [ !Equals [ !Join [ '', !Ref SecurityGroups ], ''] ]
        """
        )[0]

        cfn = Template("", template)

        with self.assertRaises(UnknownSatisfisfaction):
            cfn.conditions.satisfiable(
                {"HasSecurityGroups": True}, {"SecurityGroups": [""]}
            )

    def test_unknown_condition_satisfaction(self):
        """Test raising UnknownSatisfisfaction when condition name
        is not in conditions"""
        template = decode_str(
            """
        Parameters:
          Environment:
            Type: String
            Default: "dev"
        Conditions:
          IsProd: !Equals [!Ref Environment, "prod"]
        """
        )[0]

        cfn = Template("", template)

        # Test for line 412 in conditions.py - when condition_name
        # not in self._conditions
        with self.assertRaises(UnknownSatisfisfaction):
            cfn.conditions.satisfiable(
                {"NonExistentCondition": True}, {"Environment": "dev"}
            )

    def test_or_of_and_pairs_does_not_explode(self):
        """An Fn::Or of Fn::And pairs is encoded with a linear number of
        clauses (issue #4737). Converting it to CNF by distribution would
        create 2^30 clauses."""
        pairs = 30
        template = {
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
        }

        cfn = Template("", template)
        self.assertLess(len(cfn.conditions._cnf.data), 50 * pairs)

        self.assertListEqual(
            cfn.conditions.build_scenerios_on_region("IsDeployed", "region-0"),
            [True, False],
        )
        self.assertListEqual(
            cfn.conditions.build_scenerios_on_region("IsDeployed", "us-east-1"),
            [False],
        )
        self.assertTrue(cfn.conditions.check_implies({"IsDeployed": True}, "IsProd"))
        self.assertListEqual(
            list(cfn.conditions.build_scenarios({"IsDeployed": None, "IsProd": None})),
            [
                {"IsDeployed": True, "IsProd": True},
                {"IsDeployed": False, "IsProd": True},
                {"IsDeployed": False, "IsProd": False},
            ],
        )

    def test_ref_vs_ref_equals_joins_no_exclusion_group(self):
        """P == Q is not exclusive with P == "a" (P and Q can both be "a"),
        so only Fn::Equals against a literal exclude each other."""
        template = {
            "Parameters": {"P": {"Type": "String"}, "Q": {"Type": "String"}},
            "Conditions": {
                "IsA": {"Fn::Equals": [{"Ref": "P"}, "a"]},
                "IsB": {"Fn::Equals": [{"Ref": "P"}, "b"]},
                "Same": {"Fn::Equals": [{"Ref": "P"}, {"Ref": "Q"}]},
            },
        }
        cfn = Template("", template, ["us-east-1"])
        self.assertEqual(
            sorted(
                cfn.conditions.build_scenarios({"IsA": None, "Same": None}),
                key=lambda s: (s["IsA"], s["Same"]),
                reverse=True,
            ),
            [
                {"IsA": True, "Same": True},
                {"IsA": True, "Same": False},
                {"IsA": False, "Same": True},
                {"IsA": False, "Same": False},
            ],
        )
        self.assertFalse(cfn.conditions.check_implies({"IsA": True}, "Same"))
        self.assertEqual(
            sorted(
                cfn.conditions.build_scenarios({"IsA": None, "IsB": None}),
                key=lambda s: (s["IsA"], s["IsB"]),
                reverse=True,
            ),
            [
                {"IsA": True, "IsB": False},
                {"IsA": False, "IsB": True},
                {"IsA": False, "IsB": False},
            ],
        )
