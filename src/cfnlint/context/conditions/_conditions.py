"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

from cfnlint.conditions._boolean import (
    And,
    Boolean,
    ClauseSet,
    Not,
    Or,
    Symbol,
    add_at_most_one,
    add_constraint,
    add_equivalence,
)
from cfnlint.conditions._solver import PinSolver
from cfnlint.conditions._utils import cfn_string, get_hash
from cfnlint.context.conditions._condition import Condition, ConditionMemo
from cfnlint.context.conditions._equals import Equal
from cfnlint.context.conditions._utils import (
    build_instance_from_scenario,
    get_conditions_from_property,
)
from cfnlint.context.conditions.exceptions import Unsatisfiable

if TYPE_CHECKING:
    from cfnlint.context.context import Context, Parameter

LOGGER = logging.getLogger(__name__)


class _Pinned(Mapping):
    """The template's conditions, with the ones pinned so far replacing
    their unpinned selves.  An evolved Conditions holds one of these (a
    small dict of pins over the shared template dict) instead of a copy
    of every condition, so a lint with many resources and scenarios keeps
    O(pins) per scenario, not O(conditions)."""

    __slots__ = ("base", "pinned")

    def __init__(self, base: dict[str, Condition], pinned: dict[str, Condition]):
        self.base = base
        self.pinned = pinned  # in template order, a subset of base's keys

    def __getitem__(self, name: str) -> Condition:
        condition = self.pinned.get(name)
        return self.base[name] if condition is None else condition

    def __iter__(self) -> Iterator[str]:
        return iter(self.base)

    def __len__(self) -> int:
        return len(self.base)

    def __contains__(self, name: object) -> bool:
        return name in self.base


# evolved Conditions kept per (pins, hard pins); past this many the memo is
# emptied, bounding its memory to a few tens of MB on any template
_EVOLVED_MAX = 65536


@dataclass(frozen=True)
class Conditions:
    conditions: Mapping[str, Condition] = field(init=True, default_factory=dict)
    # The template's condition definitions and parameter constraints.
    _base_cnf: ClauseSet = field(init=True, default_factory=ClauseSet, compare=False)
    # Every pinned condition, including those assumed by scenario enumeration.
    _pins: frozenset[tuple[str, bool]] = field(
        init=True, default_factory=frozenset, compare=False
    )
    # The pins that are facts at this point in the template (resource/output
    # Condition, enclosing Fn::If).  Pins only *assumed* by condition-scenario
    # enumeration (see evolve(assumed=True)) are left out.  Reachability
    # questions (W1028) must be answered against these so that a value merely
    # being explored by enumeration is not mistaken for an impossible one.
    _hard_pins: frozenset[tuple[str, bool]] = field(
        init=True, default_factory=frozenset, compare=False
    )
    _condition_symbols: dict[str, Symbol] = field(
        init=True, default_factory=dict, compare=False
    )
    # Answers whether a set of pins is satisfiable, once per set.  It holds
    # _base_cnf, so pass it on only together with it (evolve does).
    _solver: PinSolver | None = field(
        init=True, default=None, compare=False, repr=False
    )
    _max_scenarios: int = field(init=False, default=128)
    # (pins, hard pins) -> the Conditions evolve returns for them, and
    # (name, status) -> that Condition with the status.  Both depend only on
    # the template, so the template's own Conditions keeps them for every
    # Conditions evolved from it.
    _evolved: dict[
        tuple[frozenset[tuple[str, bool]], frozenset[tuple[str, bool]]],
        "Conditions",
    ] = field(init=True, default_factory=dict, compare=False, repr=False)
    _with_status: dict[tuple[str, bool | None], Condition] = field(
        init=True, default_factory=dict, compare=False, repr=False
    )
    # The template's own statuses (normally none) and the template order of
    # the condition names; computed once on the template's Conditions and
    # shared by every Conditions evolved from it.
    _base_status: dict[str, bool] | None = field(
        init=True, default=None, compare=False, repr=False
    )
    _order: dict[str, int] | None = field(
        init=True, default=None, compare=False, repr=False
    )
    # Computed once per Conditions: every condition with a status, and the
    # scenarios enumerated for each set of condition names.
    _status_cache: dict[str, bool] | None = field(
        init=False, default=None, compare=False, repr=False
    )
    _scenario_cache: dict[tuple[str, ...], list["Conditions"]] = field(
        init=False, default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if self._solver is None:
            object.__setattr__(self, "_solver", PinSolver(self._base_cnf))

    @property
    def cnf(self) -> ClauseSet:
        """The template's condition clauses plus every pin, as a new copy."""
        return self._cnf_with(self._pins)

    @property
    def hard_cnf(self) -> ClauseSet:
        """The template's condition clauses plus the hard pins, as a new copy."""
        return self._cnf_with(self._hard_pins)

    def _cnf_with(self, pins: frozenset[tuple[str, bool]]) -> ClauseSet:
        cnf = self._base_cnf.copy()
        for name, value in sorted(pins):
            sym = self._condition_symbols[name]
            cnf.add_clause([sym if value else Not(sym)])
        return cnf

    def _is_satisfiable(self, pins: frozenset[tuple[str, bool]]) -> bool:
        assert self._solver is not None
        return self._solver.satisfiable(
            (self._condition_symbols[name], value) for name, value in pins
        )

    @classmethod
    def create_from_instance(
        cls,
        conditions: Any,
        rules: dict[str, dict],
        parameters: dict[str, "Parameter"],
    ) -> "Conditions":
        obj: dict[str, Condition] = {}
        if not isinstance(conditions, dict):
            raise ValueError("Conditions must be a object")
        memo = ConditionMemo()
        for k, v in conditions.items():
            if not isinstance(k, str):
                # Condition logical IDs must be strings (rule E8001 reports
                # non-string names).  Skip them here so the Symbol
                # table below never receives a non-string name, which would
                # raise TypeError before any rule gets to run.
                continue
            if k in memo.named:
                obj[k] = memo.named[k]
                continue
            try:
                other_conditions = conditions.copy()
                del other_conditions[k]
                fallbacks = memo.fallbacks
                obj[k] = Condition.create_from_instance(v, other_conditions, memo)
                if memo.fallbacks == fallbacks:
                    memo.named[k] = obj[k]
            except ValueError:
                obj[k] = Condition.create_from_instance(
                    {"Fn::Equals": [None, None]}, conditions
                )

        # Build the clause set with a Symbol per condition name
        cnf = ClauseSet()
        condition_symbols: dict[str, Symbol] = {}
        for name, cond in obj.items():
            sym = Symbol(name)
            condition_symbols[name] = sym
            # Add equivalence: Symbol(name) <-> condition's boolean expression
            add_equivalence(cnf, sym, cond.cnf)

        # Rules restrict the parameter values, so they constrain conditions
        # as in the template-level engine: RuleCondition -> And(Assertions).
        # A rule that uses anything but condition functions is left out.
        # Work on a copy: building an Fn::Equals sorts its list in place, and
        # W8003 reports the Rules section's own text.
        rule_exprs: list[Boolean] = []
        rule_conditions: list[Condition] = []
        for rule in deepcopy(rules).values() if isinstance(rules, dict) else []:
            try:
                parts = [
                    Condition.create_from_instance(a["Assert"], conditions, memo)
                    for a in rule["Assertions"]
                ]
                expr: Boolean = And(*[c.cnf for c in parts])
                if rule.get("RuleCondition"):
                    when = Condition.create_from_instance(
                        rule["RuleCondition"], conditions, memo
                    )
                    parts.append(when)
                    expr = Or(Not(when.cnf), expr)  # RuleCondition -> Assertions
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
            rule_conditions.extend(parts)
            rule_exprs.append(expr)

        # A condition's equals includes those of every condition it references,
        # so visit each distinct Fn::Equals once (#4984).
        distinct_equals: dict[str, Equal] = {}
        for c_v in [*obj.values(), *rule_conditions]:
            for i in c_v.equals:
                distinct_equals.setdefault(i.hash, i)

        # NAND: two Fn::Equals comparing the same single-valued reference
        # (a parameter, with or without AllowedValues, or a pseudo-parameter
        # such as AWS::Region or AWS::AccountId) to different static values
        # cannot both be true. Previously this constraint was only added
        # inside the AllowedValues loop below, so plain parameters and
        # pseudo-parameters never got it (issue #4799).
        equals_by_ref: dict[str, dict[str, list[Boolean]]] = {}
        for i in distinct_equals.values():
            if not isinstance(i.left.instance, str):
                continue
            if not isinstance(i.right.instance, dict):
                continue
            equals_by_ref.setdefault(i.right.hash, {}).setdefault(
                i.left.instance, []
            ).append(i.cnf)
        for by_value in equals_by_ref.values():
            if all(len(symbols) == 1 for symbols in by_value.values()):
                add_at_most_one(cnf, [symbols[0] for symbols in by_value.values()])
                continue
            values = list(by_value.items())
            for n, (_, symbols) in enumerate(values):
                for _, other_symbols in values[n + 1 :]:
                    for a in symbols:
                        for b in other_symbols:
                            cnf.add_clause([Not(a), Not(b)])

        # Add parameter AllowedValues constraints
        for p_k, p_v in parameters.items():
            if not p_v.allowed_values:
                continue
            allowed_values = p_v.allowed_values.copy()
            if isinstance(allowed_values, list):
                allowed_values = [cfn_string(v) for v in allowed_values]
            p_hash = get_hash({"Ref": p_k})
            equals_cnfs: list[Boolean] = []
            for i in distinct_equals.values():
                if i.right.hash == p_hash:
                    if not isinstance(i.left.instance, str):
                        continue
                    equals_cnfs.append(i.cnf)
                    if i.left.instance in allowed_values:
                        allowed_values.remove(i.left.instance)

            if not allowed_values and equals_cnfs:
                cnf.add_clause(equals_cnfs)

        for expr in rule_exprs:
            add_constraint(cnf, expr)

        solver = PinSolver(cnf)
        if rule_exprs and not solver.satisfiable(()):
            # No parameter values pass the Rules, so with them every
            # question is unsatisfiable and every branch unreachable.
            # Answer about the conditions alone instead.
            LOGGER.debug("Rules can never hold; conditions are evaluated without them")
            return cls.create_from_instance(conditions, {}, parameters)

        return cls(
            conditions=obj,
            _base_cnf=cnf,
            _condition_symbols=condition_symbols,
            _solver=solver,
        )

    def _split(self) -> tuple[dict[str, Condition], dict[str, Condition]]:
        """The template's conditions and the pins laid over them."""
        conditions = self.conditions
        if isinstance(conditions, _Pinned):
            return conditions.base, conditions.pinned
        if isinstance(conditions, dict):
            return conditions, {}
        return dict(conditions), {}

    def _template_facts(self) -> tuple[dict[str, bool], dict[str, int]]:
        if self._base_status is None:
            base, _ = self._split()
            object.__setattr__(
                self,
                "_base_status",
                {n: c.status for n, c in base.items() if c.status is not None},
            )
            object.__setattr__(self, "_order", {n: i for i, n in enumerate(base)})
        assert self._base_status is not None and self._order is not None
        return self._base_status, self._order

    def evolve(self, status: dict[str, bool], assumed: bool = False) -> "Conditions":
        """Pin one or more conditions to a value.

        When ``assumed`` is True the pin is recorded only in the pins (used for
        pruning impossible scenarios) and not in the hard pins.  This is used by
        condition-scenario enumeration, which explores both values of a
        condition; those values are hypotheses, not facts, so a branch that
        merely disagrees with the scenario currently being enumerated must not
        be reported as unreachable.
        """
        cls = self.__class__

        if not status:
            return self

        base, pinned = self._split()
        all_same = True
        for condition, condition_status in status.items():
            if condition in base:
                current = pinned.get(condition) or base[condition]
                if current.status != condition_status:
                    all_same = False
                    break
        if all_same:
            return self

        base_status, order = self._template_facts()
        new_pinned = dict(pinned)
        hard_pins = set(self._hard_pins)
        for condition, s in status.items():
            if condition not in base:
                continue
            current = new_pinned.get(condition) or base[condition]
            # Condition.evolve refuses to change a status that is set
            if current.status is not None and s != current.status:
                raise Unsatisfiable(
                    new_status=status,
                    current_status=self.status,
                )
            if s is None:
                continue
            if current.status is None:
                with_status = self._with_status.get((condition, s))
                if with_status is None:
                    with_status = base[condition].evolve(status=s)
                    self._with_status[(condition, s)] = with_status
                new_pinned[condition] = with_status
            # Only the conditions being pinned by *this* call, and only
            # when they are facts (not enumeration hypotheses), join the
            # hard pins.  Conditions carried over from a prior evolve are
            # already there if they were facts, and must stay out if they
            # were assumptions.
            if not assumed and condition in self._condition_symbols:
                hard_pins.add((condition, s))
        if len(new_pinned) > len(pinned):
            new_pinned = dict(sorted(new_pinned.items(), key=lambda kv: order[kv[0]]))

        symbols = self._condition_symbols
        pins = {
            (n, c.status)
            for n, c in new_pinned.items()
            if n in symbols and c.status is not None
        }
        pins |= {
            (n, v)
            for n, v in base_status.items()
            if n in symbols and n not in new_pinned
        }
        new_pins = frozenset(pins)

        # The result depends only on the template and the pins: every
        # condition with a status is pinned, so return the one built before
        # (it was satisfiable when it was built) without asking the solver.
        key = (new_pins, frozenset(hard_pins))
        evolved = self._evolved.get(key)
        if evolved is not None:
            return evolved
        if not self._is_satisfiable(new_pins):
            raise Unsatisfiable(
                new_status=status,
                current_status=self.status,
            )
        if len(self._evolved) >= _EVOLVED_MAX:
            self._evolved.clear()
        evolved = cls(
            conditions=_Pinned(base, new_pinned),
            _base_cnf=self._base_cnf,
            _pins=new_pins,
            _hard_pins=key[1],
            _condition_symbols=self._condition_symbols,
            _solver=self._solver,
            _evolved=self._evolved,
            _with_status=self._with_status,
            _base_status=base_status,
            _order=order,
        )
        self._evolved[key] = evolved
        return evolved

    def _status_map(self) -> dict[str, bool]:
        """Every condition with a status, in template order; computed once
        per Conditions (do not mutate).  An evolved Conditions reads its
        pin overlay, so the work is O(pins), not O(conditions)."""
        statuses = self._status_cache
        if statuses is not None:
            return statuses
        conditions = self.conditions
        if not isinstance(conditions, _Pinned):
            statuses = {
                n: c.status for n, c in conditions.items() if c.status is not None
            }
        else:
            base_status, order = self._template_facts()
            statuses = {
                n: c.status
                for n, c in conditions.pinned.items()
                if c.status is not None
            }
            if base_status:
                for n, v in base_status.items():
                    statuses.setdefault(n, v)
                statuses = dict(sorted(statuses.items(), key=lambda kv: order[kv[0]]))
        object.__setattr__(self, "_status_cache", statuses)
        return statuses

    def is_reachable(self, name: str, value: bool) -> bool:
        """Return whether ``name`` can take ``value`` given the forced facts.

        This is answered against the hard pins so that a condition merely being
        explored by scenario enumeration is not treated as impossible.  An
        unknown condition is conservatively considered reachable.
        """
        if name not in self._condition_symbols:
            return True
        return self._is_satisfiable(self._hard_pins | {(name, value)})

    def _build_conditions(self, conditions: set[str]) -> Iterator["Conditions"]:
        names = sorted(n for n in conditions if n in self._condition_symbols)
        if not names:
            yield self
            return
        assert self._solver is not None
        # The scenarios depend only on this Conditions and the names, and a
        # resource asks for the same names once per property: keep them.
        key = tuple(names)
        scenarios = self._scenario_cache.get(key)
        if scenarios is None:
            models, _ = self._solver.models(
                [(self._condition_symbols[n], v) for n, v in sorted(self._pins)],
                [self._condition_symbols[n] for n in names],
                self._max_scenarios,
            )
            scenarios = []
            for model in models:
                try:
                    scenarios.append(self.evolve(dict(zip(names, model)), assumed=True))
                except Unsatisfiable:
                    pass
            self._scenario_cache[key] = scenarios
        yield from scenarios

    def evolve_from_instance(
        self, instance: Any, context: "Context"
    ) -> Iterator[tuple[Any, "Conditions"]]:
        conditions = get_conditions_from_property(instance)

        for scenario in self._build_conditions(conditions):
            yield (
                build_instance_from_scenario(
                    instance,
                    scenario.status,
                    is_root=True,
                    context=context,
                ),
                scenario,
            )

    @property
    def status(self) -> dict[str, bool]:
        """Every condition with a status, in template order."""
        return dict(self._status_map())
