"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from cfnlint.conditions._boolean import And, Boolean, Not, Or, Symbol, true
from cfnlint.conditions._utils import get_hash
from cfnlint.context.conditions._equals import Equal
from cfnlint.helpers import FUNCTION_CONDITIONS, is_function


class ConditionMemo:
    """One Condition per name, shared by every ``Condition: name``
    reference in a template, so a condition referenced many times is
    expanded once rather than once per reference (the tree was otherwise
    exponential in the depth of the reference chain).  A definition that
    needed the ``Fn::Equals: [None, None]`` fallback anywhere below it is
    not memoised: the fallback stands for a name missing from
    ``all_conditions``, which depends on the path that led there."""

    def __init__(self) -> None:
        self.named: dict[str, "Condition"] = {}
        self.fallbacks: int = 0


@dataclass(frozen=True)
class Condition:
    instance: Any = field(init=True)
    status: bool | None = field(init=True, default=None)
    hash: str = field(init=True, default="")

    fn_equals: Equal | None = field(init=True, default=None)
    condition: list["Condition"] | "Condition" | None = field(init=True, default=None)
    cnf: Boolean = field(init=True, default=true)

    @classmethod
    def create_from_instance(
        cls,
        instance: Any,
        all_conditions: dict[str, Any],
        memo: ConditionMemo | None = None,
    ) -> "Condition":
        instance_hash = get_hash(instance)

        fn_k, fn_v = is_function(instance)
        if fn_k is None:
            raise ValueError("Condition value must be an object of length 1")
        if fn_k in FUNCTION_CONDITIONS:
            if not isinstance(fn_v, list):
                raise ValueError(f"{fn_v!r} value should be an array")
            if fn_k == "Fn::Equals":
                equal = Equal.create_from_instance(fn_v)
                return cls(
                    instance=instance,
                    fn_equals=equal,
                    cnf=equal.cnf,
                    hash=instance_hash,
                )

            condition = []
            for v in fn_v:
                condition.append(
                    Condition.create_from_instance(v, all_conditions, memo)
                )

            cnf: Boolean
            if fn_k == "Fn::And":
                cnf = And(*[c.cnf for c in condition])
            elif fn_k == "Fn::Or":
                cnf = Or(*[c.cnf for c in condition])
            else:  # Fn::Not
                if len(condition) != 1:
                    raise ValueError(
                        f"Fn::Not expects only one condition, got {len(condition)}"
                    )
                cnf = Not(condition[0].cnf)

            return cls(
                instance=instance, condition=condition, cnf=cnf, hash=instance_hash
            )

        if fn_k == "Condition":
            if not isinstance(fn_v, str):
                raise ValueError(f"Condition value {fn_v!r} must be a string")
            # Build the sub-condition for is_region/equals traversal
            if memo is not None and fn_v in memo.named:
                c = memo.named[fn_v]
            else:
                sub_condition = all_conditions.get(fn_v)
                fallbacks = memo.fallbacks if memo is not None else 0
                try:
                    sub_all_conditions = all_conditions.copy()
                    del sub_all_conditions[fn_v]
                    c = Condition.create_from_instance(
                        sub_condition, sub_all_conditions, memo
                    )
                    if memo is not None and memo.fallbacks == fallbacks:
                        memo.named[fn_v] = c
                except Exception:
                    if memo is not None:
                        memo.fallbacks += 1
                    c = Condition.create_from_instance(
                        {"Fn::Equals": [None, None]}, all_conditions
                    )
            # Settle the cached walks now, while the chain below is built:
            # each referenced condition has its own settled, so this costs
            # one level.  Read first from the top of a long chain written
            # top-down, the walks would recurse through every link.
            c.equals
            c.is_region
            # Use a Symbol for the condition name instead of the expanded CNF
            # The equivalence constraint is added in Conditions.create_from_instance
            return cls(
                instance=instance,
                condition=c,
                cnf=Symbol(fn_v),
                hash=instance_hash,
            )

        raise ValueError(f"Unknown key {fn_k!r} in condition")

    def evolve(self, status: bool | None) -> "Condition":
        cls = self.__class__

        if self.status is not None:
            if status != self.status:
                raise ValueError(f"Resetting status to {status} from {self.status}")

        return cls(
            instance=self.instance,
            status=status,
            cnf=self.cnf,
            hash=self.hash,  # Pass the hash along
        )

    @cached_property
    def is_region(self) -> bool:
        """Returns True or False if the condition is based on region

        Args: None

        Returns:
            bool
                Returns True or False if the condition is based on region
        """
        if self.fn_equals:
            return self.fn_equals.is_region
        if isinstance(self.condition, list):
            for c in self.condition:
                if c.is_region:
                    return True
            return False
        if self.condition:
            return self.condition.is_region
        return False

    @cached_property
    def equals(self) -> list[Equal]:
        """Returns a Sequence of the Equals that make up the Condition

        Args: None

        Returns:
            Sequence[EqualParameter] | Sequence[Equal] | None:
                The Equal that are part of the condition, one per distinct
                Fn::Equals
        """
        if self.fn_equals:
            return [self.fn_equals]
        if isinstance(self.condition, list):
            equals: dict[str, Equal] = {}
            for c in self.condition:
                for equal in c.equals:
                    equals.setdefault(equal.hash, equal)
            return list(equals.values())
        if self.condition:
            return self.condition.equals
        return []
