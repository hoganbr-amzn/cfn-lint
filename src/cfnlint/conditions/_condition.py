"""
Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Union

from cfnlint.conditions._boolean import And, Boolean, BooleanTrue, Not, Or, Symbol
from cfnlint.conditions._equals import Equal
from cfnlint.helpers import FUNCTION_CONDITIONS

# we leave the type hinting here
_CONDITION = Dict[str, Union[str, Sequence["_CONDITION"]]]


class ConditionMemo:
    """One ConditionNamed per name, shared by every ``Condition: name``
    reference, so a condition referenced many times is parsed once instead
    of once per reference (which made the tree exponential in the depth of
    the reference chain).  A name that is being built when it is referenced
    again is a cycle and raises, as the unbounded recursion did before."""

    def __init__(self) -> None:
        self.named: dict[str, ConditionNamed] = {}
        self.building: set[str] = set()


class Condition:
    """The generic class to represent any type of Condition"""

    def __init__(self) -> None:
        self._fn_equals: Equal | None = None
        self._condition: ConditionList | ConditionNamed | None = None

    def _init_condition(
        self,
        condition: _CONDITION,
        all_conditions: dict[str, dict],
        memo: ConditionMemo | None = None,
    ) -> None:
        if len(condition) == 1:
            for k, v in condition.items():
                if k in FUNCTION_CONDITIONS:
                    if not isinstance(v, list):
                        raise ValueError(f"{k} value should be an array")
                    if k == "Fn::Equals":
                        self._fn_equals = Equal(v)
                    elif k == "Fn::And":
                        self._condition = ConditionAnd(v, all_conditions, memo)
                    elif k == "Fn::Or":
                        self._condition = ConditionOr(v, all_conditions, memo)
                    elif k == "Fn::Not":
                        self._condition = ConditionNot(v, all_conditions, memo)
                elif k == "Condition":
                    if not isinstance(v, str):
                        raise ValueError(f"Condition value {v!r} must be a string")
                    self._condition = ConditionNamed.named(v, all_conditions, memo)
                else:
                    raise ValueError(f"Unknown key ({k}) in condition")
        else:
            raise ValueError("Condition value must be an object of length 1")

    @property
    def equals(self) -> list[Equal]:
        """Returns a Sequence of the Equals that make up the Condition

        Args: None

        Returns:
            Sequence[EqualParameter] | Sequence[Equal] | None:
                The Equal that are part of the condition
        """
        if self._fn_equals:
            return [self._fn_equals]
        if self._condition:
            return self._condition.equals
        return []

    def build_cnf(self, params: dict[str, Symbol]) -> Boolean:
        """Build the boolean expression of the condition over ``params``

        Args:
            params dict[str, Symbol]: the hash of an Equal, or a condition
                    name, to the Symbol or constant standing for it

        Returns:
            Boolean: the expression
        """
        if self._condition:
            return self._condition.build_cnf(params)
        if self._fn_equals:
            return self._fn_equals.build_cnf(params)
        return BooleanTrue()

    def _test(self, scenarios: Mapping[str, str]) -> bool:
        if self._fn_equals:
            return self._fn_equals.test(scenarios)
        if self._condition:
            # pylint: disable=W0212
            return self._condition._test(scenarios)
        return False


class ConditionList(Condition):
    """The generic class to represent any type of List condition
    List conditions are And, Or, Not
    """

    def __init__(
        self,
        conditions: Sequence[_CONDITION],
        all_conditions: dict[str, dict],
        memo: ConditionMemo | None = None,
    ) -> None:
        super().__init__()
        self._conditions: list[ConditionUnnammed] = []
        self._prefix_path: str = ""
        for condition in conditions:
            self._conditions.append(ConditionUnnammed(condition, all_conditions, memo))

    @property
    def equals(self) -> list[Equal]:
        """Returns a List of the Equals that make up the Condition

        Args: None

        Returns:
            list[Equal]: The Equal that are part of the condition
        """
        # One entry per distinct Fn::Equals: a condition referenced from
        # several places contributes its equals once.
        equals: dict[str, Equal] = {}
        for condition in self._conditions:
            for equal in condition.equals:
                equals.setdefault(equal.hash, equal)
        return list(equals.values())


class ConditionAnd(ConditionList):
    """Represents the logic specific to an And Condition"""

    def __init__(
        self,
        conditions: Sequence[_CONDITION],
        all_conditions: dict[str, dict],
        memo: ConditionMemo | None = None,
    ) -> None:
        super().__init__(conditions, all_conditions, memo)
        self._prefix_path = "Fn::And"

    def build_cnf(self, params: dict[str, Symbol]) -> Boolean:
        """Build the And of the child conditions over ``params``
        Args:
            params dict[str, Symbol]: the hash of an Equal, or a condition
                    name, to the Symbol or constant standing for it
        Returns:
            Boolean: the And, or the single child or a constant it folds to
        """
        conditions: list[Any] = []
        for child in self._conditions:
            conditions.append(child.build_cnf(params))

        return And(*conditions)

    def _test(self, scenarios: Mapping[str, str]) -> bool:
        # pylint: disable=W0212
        return all(condition._test(scenarios) for condition in self._conditions)


class ConditionNot(ConditionList):
    """Represents the logic specific to an Not Condition"""

    def __init__(
        self,
        conditions: Sequence[_CONDITION],
        all_conditions: dict[str, dict],
        memo: ConditionMemo | None = None,
    ) -> None:
        super().__init__(conditions, all_conditions, memo)
        self._prefix_path = "Fn::Not"
        if len(conditions) != 1:
            raise ValueError("Condition length must be 1")

    def build_cnf(self, params: dict[str, Symbol]) -> Boolean:
        """Build the negation of the child condition over ``params``
        Args:
            params dict[str, Symbol]: the hash of an Equal, or a condition
                    name, to the Symbol or constant standing for it
        Returns:
            Boolean: the Not, or the constant it folds to
        """
        return Not(self._conditions[0].build_cnf(params))

    def _test(self, scenarios: Mapping[str, str]) -> bool:
        # pylint: disable=W0212
        return not any(condition._test(scenarios) for condition in self._conditions)


class ConditionOr(ConditionList):
    """Represents the logic specific to an Or Condition"""

    def __init__(
        self,
        conditions: Sequence[_CONDITION],
        all_conditions: dict[str, dict],
        memo: ConditionMemo | None = None,
    ) -> None:
        super().__init__(conditions, all_conditions, memo)
        self._prefix_path = "Fn::Or"

    def build_cnf(self, params: dict[str, Symbol]) -> Boolean:
        """Build the Or of the child conditions over ``params``
        Args:
            params dict[str, Symbol]: the hash of an Equal, or a condition
                    name, to the Symbol or constant standing for it
        Returns:
            Boolean: the Or, or the single child or a constant it folds to
        """
        conditions: list[Any] = []
        for child in self._conditions:
            conditions.append(child.build_cnf(params))
        return Or(*conditions)

    def _test(self, scenarios: Mapping[str, str]) -> bool:
        # pylint: disable=W0212
        return any(condition._test(scenarios) for condition in self._conditions)


class ConditionUnnammed(Condition):
    """Represents an unnamed condition which is basically a nested Equals"""

    def __init__(
        self,
        condition: Any,
        all_conditions: dict[str, dict],
        memo: ConditionMemo | None = None,
    ) -> None:
        super().__init__()
        if isinstance(condition, dict):
            self._init_condition(condition, all_conditions, memo)
        else:
            raise ValueError("Condition must have a value that is an object")


class ConditionNamed(Condition):
    """The parent condition that directly represents a named condition in a template"""

    def __init__(
        self,
        name: str,
        all_conditions: dict[str, dict],
        memo: ConditionMemo | None = None,
    ) -> None:
        super().__init__()
        self._equals: list[Equal] | None = None
        condition = all_conditions.get(name)
        if not isinstance(condition, dict):
            raise ValueError(f"Condition {name} must have a value that is an object")
        self._name = name
        if memo is None:
            self._init_condition(condition, all_conditions, memo)
            return
        if name in memo.building:
            raise ValueError(f"Condition {name} refers to itself")
        memo.building.add(name)
        try:
            self._init_condition(condition, all_conditions, memo)
        finally:
            memo.building.discard(name)
        memo.named[name] = self

    @classmethod
    def named(
        cls, name: str, all_conditions: dict[str, dict], memo: ConditionMemo | None
    ) -> "ConditionNamed":
        """The ConditionNamed for ``name``, built once per memo."""
        if memo is not None and name in memo.named:
            return memo.named[name]
        return cls(name, all_conditions, memo)

    @property
    def equals(self) -> list[Equal]:
        if self._equals is None:
            self._equals = super().equals
        return self._equals

    def build_cnf(self, params: dict[str, Symbol]) -> Boolean:
        """The Symbol standing for this condition when ``params`` has one,
        otherwise its expression."""
        if self._name in params:
            return params[self._name]
        return super().build_cnf(params)

    def test(self, scenarios: Mapping[str, str]) -> bool:
        """Test a condition based on a scenario"""
        return self._test(scenarios)
