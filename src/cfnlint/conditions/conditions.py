"""
Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

from __future__ import annotations

import logging
import traceback
from functools import lru_cache
from typing import Any, Iterable, Iterator, Set, Tuple

from sympy import Not, Symbol
from sympy.assumptions.cnf import EncodedCNF
from sympy.logic.boolalg import BooleanFalse, BooleanTrue

from cfnlint.conditions._condition import ConditionNamed, _Memo
from cfnlint.conditions._equals import Equal, EqualParameter
from cfnlint.conditions._errors import UnknownSatisfisfaction
from cfnlint.conditions._rule import Rule
from cfnlint.conditions._solver import PinSolver
from cfnlint.conditions._utils import (
    add_at_most_one,
    add_constraint,
    add_equivalence,
    get_hash,
)
from cfnlint.helpers import get_partition

LOGGER = logging.getLogger(__name__)


class Conditions:
    """Conditions provides the logic for relating individual condition together"""

    _max_scenarios: int = 128  # equivalent to 2^7

    def __init__(self, cfn) -> None:
        self._conditions: dict[str, ConditionNamed] = {}
        self._parameters: dict[str, list[str]] = {}
        self._rules: list[Rule] = []
        self._memo = _Memo()
        self._init_conditions(cfn=cfn)
        self._init_parameters(cfn=cfn)
        self._init_rules(cfn=cfn)
        self._cnf, self._solver_params = self._build_cnf(list(self._conditions.keys()))
        self._solver = PinSolver(self._cnf)

    def _init_conditions(self, cfn):
        conditions = cfn.template.get("Conditions")
        if isinstance(conditions, dict):
            for k, _ in conditions.items():
                try:
                    self._conditions[k] = ConditionNamed.named(
                        k, conditions, self._memo
                    )
                except ValueError as e:
                    LOGGER.debug(
                        "Captured error while building condition %s: %s", k, str(e)
                    )
                except Exception as e:  # pylint: disable=broad-exception-caught
                    if LOGGER.getEffectiveLevel() == logging.DEBUG:
                        error_message = traceback.format_exc()
                    else:
                        error_message = str(e)
                    LOGGER.debug(
                        "Captured unknown error while building condition %s: %s",
                        k,
                        error_message,
                    )

    def _init_parameters(self, cfn: Any) -> None:
        parameters = cfn.template.get("Parameters")
        if not isinstance(parameters, dict):
            return
        for parameter_name, parameter in parameters.items():
            if not isinstance(parameter, dict):
                continue
            allowed_values = parameter.get("AllowedValues")
            if not allowed_values or not isinstance(allowed_values, list):
                continue

            param_hash = get_hash({"Ref": parameter_name})
            self._parameters[param_hash] = []
            for allowed_value in allowed_values:
                if isinstance(allowed_value, (str, int, float, bool)):
                    self._parameters[param_hash].append(get_hash(str(allowed_value)))

    def _init_rules(self, cfn: Any) -> None:
        rules = cfn.template.get("Rules")
        conditions = cfn.template.get("Conditions", {})
        if not isinstance(rules, dict) or not isinstance(conditions, dict):
            return
        for k, v in rules.items():
            if not isinstance(v, dict):
                continue
            try:
                self._rules.append(Rule(v, conditions, self._memo))
            except ValueError as e:
                LOGGER.debug("Captured error while building rule %s: %s", k, str(e))
            except Exception as e:  # pylint: disable=broad-exception-caught
                if LOGGER.getEffectiveLevel() == logging.DEBUG:
                    error_message = traceback.format_exc()
                else:
                    error_message = str(e)
                LOGGER.debug(
                    "Captured unknown error while building rule %s: %s",
                    k,
                    error_message,
                )

    def get(self, name: str, default: Any = None) -> ConditionNamed:
        """Return the conditions"""
        return self._conditions.get(name, default)

    def _satisfiable(self, pins: Iterable[tuple[Any, bool]]) -> bool:
        """Whether the template's conditions allow ``pins``: pairs of a
        solver variable (a condition's Symbol from ``_solver_params`` or an
        Fn::Equals Symbol) and the value it takes.  A static Fn::Equals is a
        constant rather than a Symbol: pinning it to its value says nothing
        and pinning it to the other value is impossible."""
        symbols = []
        for literal, value in pins:
            if isinstance(literal, (BooleanTrue, BooleanFalse)):
                if bool(literal) != value:
                    return False
                continue
            symbols.append((literal, value))
        return self._solver.satisfiable(symbols)

    def _build_cnf(
        self, condition_names: list[str]
    ) -> Tuple[EncodedCNF, dict[str, Any]]:
        cnf = EncodedCNF()

        # build parameters and equals into solver
        equal_vars: dict[str, Symbol | BooleanFalse | BooleanTrue] = {}

        equals: dict[str, Equal] = {}

        # Fn::Equals hash -> hashes of the parameters it compares
        by_parameter: dict[str, list[str]] = {}

        def _build_equal_vars(c_equals: list[Equal]):
            for c_equal in c_equals:
                if c_equal.hash in equal_vars:
                    continue

                if c_equal.is_static is not None:
                    # A static Fn::Equals (two equal literals, or Ref P
                    # against Ref P) has the same value whatever P is, so it
                    # says nothing about P and joins no exclusion group.
                    if c_equal.is_static:
                        equal_vars[c_equal.hash] = BooleanTrue()
                    else:
                        equal_vars[c_equal.hash] = BooleanFalse()
                else:
                    equal_vars[c_equal.hash] = Symbol(c_equal.hash)
                    for param in c_equal.parameters:
                        by_parameter.setdefault(param.hash, []).append(c_equal.hash)
                equals[c_equal.hash] = c_equal

        for rule in self._rules:
            _build_equal_vars(rule.equals)

        for condition_name in condition_names:
            _build_equal_vars(self._conditions[condition_name].equals)

        # Two Fn::Equals that compare the same parameter cannot both be
        # true (NAND).
        for hashes in by_parameter.values():
            add_at_most_one(cnf, [equal_vars[h] for h in hashes])

        # Determine if a set of conditions can never be all false
        allowed_values = {k: list(v) for k, v in self._parameters.items()}
        if allowed_values:
            # iteration 1 cleans up all the hash values
            # from allowed_values to know if we
            # used them all
            for _, equal_1 in equals.items():
                for param in equal_1.parameters:
                    if param.hash not in allowed_values:
                        continue
                    if isinstance(equal_1.left, str):
                        if get_hash(equal_1.left) in allowed_values[param.hash]:
                            allowed_values[param.hash].remove(get_hash(equal_1.left))
                        else:
                            equal_vars[equal_1.hash] = BooleanFalse()
                    if isinstance(equal_1.right, str):
                        if get_hash(equal_1.right) in allowed_values[param.hash]:
                            allowed_values[param.hash].remove(get_hash(equal_1.right))
                        else:
                            equal_vars[equal_1.hash] = BooleanFalse()

            # iteration 2 builds the cnf formulas to make sure any empty lists
            # are now full not equals
            for allowed_hash, allowed_value in allowed_values.items():
                # means the list is empty and all allowed values are validated
                # so not all equals can be false
                if not allowed_value:
                    prop = None
                    for _, equal_1 in equals.items():
                        for param in equal_1.parameters:
                            if allowed_hash == param.hash:
                                if prop is None:
                                    prop = Not(equal_vars[equal_1.hash])
                                else:
                                    prop = prop & Not(equal_vars[equal_1.hash])
                    # Need to make sure they aren't all False
                    # So Not(Not(Equal1) & Not(Equal2))
                    # When Equal1 False and Equal2 False
                    # Not(True & True) = False allowing this not to happen
                    if prop is not None:
                        cnf.add_prop(Not(prop))

        # Create a Symbol per condition name and add implication constraints
        # so that build_true_cnf/build_false_cnf can use the simple Symbol
        # instead of the full boolean expression (avoids CNF explosion).
        # We add condition symbols to equal_vars FIRST so that when
        # build_cnf is called, nested !Condition references resolve to
        # the simple Symbol instead of expanding the full expression.
        cond_symbols: dict[str, Symbol] = {}
        for condition_name in condition_names:
            cond_symbols[condition_name] = Symbol(f"__cond_{condition_name}")

        # Build equivalences using cond_symbols for nested condition refs
        # but NOT for the condition being defined (to avoid circularity)
        for condition_name in condition_names:
            cond_sym = cond_symbols[condition_name]
            # Merge equal_vars with other condition symbols (excluding self)
            build_params = dict(equal_vars)
            for other_name, other_sym in cond_symbols.items():
                if other_name != condition_name:
                    build_params[other_name] = other_sym
            cond_expr = self._conditions[condition_name].build_cnf(build_params)
            add_equivalence(cnf, cond_sym, cond_expr)

        equal_vars.update(cond_symbols)

        for rule in self._rules:
            add_constraint(cnf, rule.build_cnf(equal_vars))

        return (cnf, equal_vars)

    def build_scenarios(
        self, conditions: dict[str, Set[bool]], region: str | None = None
    ) -> Iterator[dict[str, bool]]:
        """Given a list of condition names this function will
        yield scenarios that represent those conditions and
        there result (True/False)

        Args:
            condition_names (list[str]): A list of condition names

        Returns:
            Iterator[dict[str, bool]]: yield dict objects of {ConditionName: True/False}
        """
        # nothing to yield if there are no conditions
        if len(conditions) == 0:
            return

        fixed_pins: list[tuple[Any, bool]] = []
        condition_names = []
        conditions_set = {}
        for condition_name, values in conditions.items():
            if condition_name in self._conditions:
                if values == {True}:
                    fixed_pins.append((self._solver_params[condition_name], True))
                    conditions_set[condition_name] = True
                    continue
                if values == {False}:
                    fixed_pins.append((self._solver_params[condition_name], False))
                    conditions_set[condition_name] = False
                    continue
            condition_names.append(condition_name)
        # Callers collect names in sets; sort them so the scenarios tried
        # before the cap, and their order, do not depend on the hash seed.
        condition_names.sort(key=str)

        pins = list(fixed_pins)
        if region:
            # A region fixes every Fn::Equals on AWS::Region and AWS::Partition
            partition = get_partition(region)
            for condition in self._conditions.values():
                for eql in condition.equals:
                    is_region, equal_region = eql.is_region
                    if is_region:
                        pins.append(
                            (self._solver_params[eql.hash], equal_region == region)
                        )
                    is_partition, equal_partition = eql.is_partition
                    if is_partition:
                        pins.append(
                            (
                                self._solver_params[eql.hash],
                                equal_partition == partition,
                            )
                        )
        try:
            symbols = []
            for condition_name in condition_names:
                # KeyError for a condition name the template does not define
                self._conditions[condition_name]
                symbols.append(self._solver_params[condition_name])
        except KeyError:
            return
        solver_pins = []
        for literal, value in pins:
            if isinstance(literal, (BooleanTrue, BooleanFalse)):
                if bool(literal) != value:
                    return
                continue
            solver_pins.append((literal, value))
        # Enumerate the satisfiable scenarios only, in a deterministic order,
        # at most _max_scenarios of them; truncation is counted on the solver.
        models, truncated = self._solver.models(
            solver_pins, symbols, self._max_scenarios
        )
        if truncated:
            LOGGER.debug(
                "Condition scenarios truncated at %s for %s",
                self._max_scenarios,
                condition_names,
            )
        for model in models:
            yield {**dict(zip(condition_names, model)), **conditions_set}

    def check_implies(self, scenarios: dict[str, bool], implies: str) -> bool:
        """Based on a bunch of scenario conditions and their Truth/False value
        determine if implies condition is True any time the scenarios are satisfied
        solver, solver_params = self._build_solver(list(scenarios.keys()) + [implies])

        Args:
            scenarios (dict[str, bool]): A list of condition names
            and if they are True or False implies: the condition name that
            we are implying will also be True

        Returns:
            bool: if the implied condition will be True if the scenario is True
        """
        try:
            # if the implies condition has to be false in the scenarios we
            # know it can never be true
            if not scenarios.get(implies, True):
                return False

            # The scenario implies the condition when the scenario together
            # with the condition being false cannot be satisfied.
            pins: list[tuple[Any, bool]] = []
            for condition_name, opt in scenarios.items():
                self._conditions[condition_name]
                pins.append((self._solver_params[condition_name], bool(opt)))
            self._conditions[implies]
            pins.append((self._solver_params[implies], False))
            return not self._satisfiable(pins)
        except KeyError:
            # KeyError is because the listed condition doesn't exist because of bad
            #  formatting or just the wrong condition name
            return True

    @lru_cache
    def is_always_true(self, condition_name: str) -> bool:
        """Determine if a condition provably always evaluates to true.

        Args:
            condition_name (str): The name of the condition to validate

        Returns:
            bool: True when the condition cannot evaluate to False,
                otherwise False
        """
        if condition_name not in self._conditions:
            return False
        return self.check_implies({}, condition_name)

    @lru_cache
    def build_scenerios_on_region(self, condition_name: str, region: str) -> list[bool]:
        """Based on a region validate if the condition_name could be true

        Args:
            condition_name (str): The name of the condition we are validating against
            region (str): the name of the region

        Returns:
            list[bool]: Returns True, False, or True and False depending on if the
               condition could be True, False or both based on the region parameter
        """
        region_pins: list[tuple[Any, bool]] = []
        found_pseudo = False

        # validate the condition name exists
        # return True/False if it doesn't
        if condition_name not in self._conditions:
            return [True, False]

        # A region fixes the partition too, so an Equals against
        # Ref AWS::Partition is decided the same way as one against Ref AWS::Region
        partition = get_partition(region)
        for eql in self._conditions[condition_name].equals:
            is_region, equal_region = eql.is_region
            if is_region:
                found_pseudo = True
                region_pins.append(
                    (self._solver_params[eql.hash], equal_region == region)
                )
            is_partition, equal_partition = eql.is_partition
            if is_partition:
                found_pseudo = True
                region_pins.append(
                    (self._solver_params[eql.hash], equal_partition == partition)
                )

        # The condition doesn't use AWS::Region or AWS::Partition so it can be
        # True or False
        # Note: It is possible its a hard coded condition but
        # for now we will return True and False
        if not found_pseudo:
            return [True, False]

        results = []
        condition = self._solver_params[condition_name]
        if self._satisfiable(region_pins + [(condition, True)]):
            results.append(True)
        if self._satisfiable(region_pins + [(condition, False)]):
            results.append(False)

        return results

    def satisfiable(
        self, conditions: dict[str, bool], parameter_values: dict[str, str]
    ) -> bool:
        """Given a list of condition names this function will
        determine if the conditions are satisfied

        Args:
            condition_names (dict[str, bool]): A list of condition names with if
              they are True or False

        Returns:
            bool: True if the conditions are satisfied

        Raises:
            UnknownSatisfisfaction: If we don't know how to satisfy a condition
        """
        if not conditions:
            if self._rules:
                return self._satisfiable([])
            else:
                return True

        pins: list[tuple[Any, bool]] = []
        at_least_one_param_found = False

        for condition_name, opt in conditions.items():
            if condition_name not in self._conditions:
                raise UnknownSatisfisfaction(
                    f"Can't resolve satisfaction for {condition_name!r}"
                )
            for c_equals in self._conditions[condition_name].equals:
                found_params = {}
                for param, value in parameter_values.items():
                    ref_hash = get_hash({"Ref": param})

                    for c_equal_param in c_equals.parameters:
                        if isinstance(c_equal_param, EqualParameter):
                            if c_equal_param.satisfiable is False:
                                raise UnknownSatisfisfaction(
                                    f"Can't resolve satisfaction for {condition_name!r}"
                                )

                    if ref_hash in c_equals.parameters:
                        found_params = {ref_hash: value}

                if not found_params:
                    continue

                at_least_one_param_found = True
                # Pin the Fn::Equals the template encodes. When it is a
                # constant (static, or false because the value is not in the
                # parameter's AllowedValues) a value that disagrees with it
                # is unsatisfiable rather than ignored.
                pins.append(
                    (self._solver_params[c_equals.hash], c_equals.test(found_params))
                )
                pins.append((self._solver_params[condition_name], bool(opt)))

        if at_least_one_param_found is False:
            if self._rules:
                return self._satisfiable([])
            else:
                return True

        return self._satisfiable(pins)
