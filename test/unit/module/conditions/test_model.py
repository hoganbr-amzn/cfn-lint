"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

from unittest import TestCase

from cfnlint.conditions._boolean import BooleanTrue, Symbol
from cfnlint.conditions._model import PointModel, fold

MAPPINGS = {
    "Envs": {
        "prod": {"Bucket": "prod-bucket", "Count": 3, "Public": False},
        "dev": {"Bucket": "dev-bucket", "Tags": ["a", "b"]},
    }
}


class TestFold(TestCase):
    """fold writes an Fn::Equals operand that needs no parameter as the
    string CloudFormation compares, and returns None for any other."""

    def test_scalars_fold_as_cloudformation_strings(self):
        for value, expected in [
            ("x", "x"),
            (1, "1"),
            (1.5, "1.5"),
            (True, "true"),
            (False, "false"),
        ]:
            with self.subTest(value=value):
                self.assertEqual(fold(value, MAPPINGS), expected)

    def test_select_folds_to_the_indexed_element(self):
        self.assertEqual(fold({"Fn::Select": ["1", ["dev", "prod"]]}, None), "prod")
        self.assertEqual(fold({"Fn::Select": [0, ["dev", "prod"]]}, None), "dev")
        # the index may itself fold
        self.assertEqual(
            fold({"Fn::Select": [{"Fn::Select": [0, ["1"]]}, ["dev", "prod"]]}, None),
            "prod",
        )

    def test_select_with_a_bad_index_does_not_fold(self):
        for index in ["-1", "1_0", "2", "x", "١", {"Ref": "I"}, 1.0]:
            with self.subTest(index=index):
                self.assertIsNone(fold({"Fn::Select": [index, ["dev", "prod"]]}, None))
        self.assertIsNone(fold({"Fn::Select": ["0", {"Ref": "List"}]}, None))
        self.assertIsNone(fold({"Fn::Select": ["0"]}, None))

    def test_find_in_map_folds_to_the_mapping_value(self):
        self.assertEqual(
            fold({"Fn::FindInMap": ["Envs", "prod", "Bucket"]}, MAPPINGS),
            "prod-bucket",
        )
        # a number or boolean value is written as CloudFormation compares it
        self.assertEqual(
            fold({"Fn::FindInMap": ["Envs", "prod", "Count"]}, MAPPINGS), "3"
        )
        self.assertEqual(
            fold({"Fn::FindInMap": ["Envs", "prod", "Public"]}, MAPPINGS), "false"
        )
        # keys may themselves fold
        self.assertEqual(
            fold(
                {
                    "Fn::FindInMap": [
                        "Envs",
                        {"Fn::Select": [1, ["dev", "prod"]]},
                        "Bucket",
                    ]
                },
                MAPPINGS,
            ),
            "prod-bucket",
        )

    def test_find_in_map_miss_does_not_fold(self):
        for args in [
            ["Envs", "prod", "Missing"],
            ["Envs", "staging", "Bucket"],
            ["Other", "prod", "Bucket"],
            ["Envs", {"Ref": "Env"}, "Bucket"],
            ["Envs", "dev", "Tags"],  # a list value
            ["Envs", "prod"],
        ]:
            with self.subTest(args=args):
                self.assertIsNone(fold({"Fn::FindInMap": args}, MAPPINGS))
        self.assertIsNone(fold({"Fn::FindInMap": ["Envs", "prod", "Bucket"]}, None))

    def test_join_folds_when_every_part_does(self):
        self.assertEqual(fold({"Fn::Join": ["-", ["a", 1, True]]}, None), "a-1-true")
        self.assertEqual(
            fold(
                {
                    "Fn::Join": [
                        "",
                        ["pre-", {"Fn::FindInMap": ["Envs", "dev", "Bucket"]}],
                    ]
                },
                MAPPINGS,
            ),
            "pre-dev-bucket",
        )

    def test_join_with_an_unfoldable_part_does_not_fold(self):
        self.assertIsNone(fold({"Fn::Join": ["-", ["a", {"Ref": "P"}]]}, None))
        self.assertIsNone(fold({"Fn::Join": [["-"], ["a", "b"]]}, None))
        self.assertIsNone(fold({"Fn::Join": ["-", "ab"]}, None))

    def test_other_values_do_not_fold(self):
        for value in [
            {"Ref": "P"},
            {"Fn::Sub": "a"},
            {"Fn::If": ["C", "a", "b"]},
            {"Fn::Select": ["0", ["a"]], "Ref": "P"},
            ["a"],
            None,
        ]:
            with self.subTest(value=value):
                self.assertIsNone(fold(value, MAPPINGS))


class TestPointModel(TestCase):
    def test_add_equal_records_literals_and_references(self):
        model = PointModel()
        model.add_equal(Symbol("a"), ({"Ref": "P"}, "p"), ("x", ""), None)
        model.add_equal(
            Symbol("b"),
            ({"Fn::Select": ["1", ["dev", "prod"]]}, "s"),
            ({"Ref": "Q"}, "q"),
            None,
        )
        # a static Fn::Equals is a constant, not an atom
        model.add_equal(BooleanTrue(), ("x", ""), ("x", ""), None)
        self.assertEqual(
            model.atoms,
            {
                Symbol("a"): (("R", "p"), ("L", "x")),
                Symbol("b"): (("L", "prod"), ("R", "q")),
            },
        )

    def test_add_domain_keeps_scalar_allowed_values_only(self):
        model = PointModel()
        model.add_domain("p", ["a", 1, True])
        model.add_domain("q", ["a", ["b"]])
        model.add_domain("r", [])
        model.add_domain("s", "a")
        self.assertEqual(model.domains, {"p": ["a", "1", "true"]})
