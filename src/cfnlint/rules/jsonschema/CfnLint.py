"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

from cfnlint.jsonschema._utils import Unset
from cfnlint.rules import CloudFormationLintRule


class CfnLint(CloudFormationLintRule):
    id = "E1101"
    shortdesc = "Validate an item against additional checks"
    description = "Use supplemental logic to validate properties against"
    tags = []

    def __init__(self) -> None:
        super().__init__()
        self._rules_by_keyword: dict[str, list[CloudFormationLintRule]] = {}
        self._indexed_rules: tuple[CloudFormationLintRule | None, ...] = ()

    def _rules_for_keyword(self, keyword: str) -> list[CloudFormationLintRule]:
        # child rules are linked after construction and can change
        # between runs so rebuild the index when they do
        # rules compare equal by id so compare the instances themselves
        rules = tuple(self.child_rules.values())
        if len(rules) != len(self._indexed_rules) or any(
            rule is not indexed for rule, indexed in zip(rules, self._indexed_rules)
        ):
            index: dict[str, list[CloudFormationLintRule]] = {}
            for rule in rules:
                if rule is None or not rule.id:
                    continue
                for rule_keyword in getattr(rule, "keywords", []):
                    index.setdefault(rule_keyword, []).append(rule)
            self._rules_by_keyword = index
            self._indexed_rules = rules
        return self._rules_by_keyword.get(keyword, [])

    # pylint: disable=unused-argument
    def cfnLint(self, validator, keywords, instance, schema):
        validator = validator.evolve(
            function_filter=validator.function_filter.evolve(
                add_cfn_lint_keyword=False,
            )
        )

        for keyword in keywords:
            for rule in self._rules_for_keyword(keyword):
                for err in rule.validate(validator, keyword, instance, schema):
                    if err.rule is None:
                        if isinstance(err.validator, Unset):
                            err.rule = rule
                    yield err
