"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

from sympy import Dummy, Not, Or, Symbol
from sympy.assumptions.cnf import EncodedCNF
from sympy.logic.inference import satisfiable

from cfnlint.conditions._utils import add_equivalence, copy_cnf, get_hash
from cfnlint.context.conditions._condition import Condition
from cfnlint.context.conditions._utils import (
    build_instance_from_scenario,
    get_conditions_from_property,
)
from cfnlint.context.conditions.exceptions import Unsatisfiable

if TYPE_CHECKING:
    from cfnlint.context.context import Context, Parameter

# A set of pinned conditions: (condition name, pinned value) pairs.
Pins = frozenset[tuple[str, bool]]


def _is_satisfiable(cnf: EncodedCNF) -> bool:
    # satisfiable() answers an empty CNF with an empty model, which reads
    # as False; an empty CNF has nothing to violate.
    return not cnf.data or bool(satisfiable(cnf))


def _has_cycle(references: dict[str, set[str]]) -> bool:
    """Return whether a condition refers to itself, directly or through
    other conditions (a topological sort that cannot finish)."""
    pending = {
        name: {ref for ref in refs if ref in references}
        for name, refs in references.items()
    }
    referrers: dict[str, list[str]] = {}
    for name, refs in pending.items():
        for ref in refs:
            referrers.setdefault(ref, []).append(name)
    ready = [name for name, refs in pending.items() if not refs]
    while ready:
        name = ready.pop()
        del pending[name]
        for referrer in referrers.get(name, ()):
            pending[referrer].discard(name)
            if not pending[referrer]:
                ready.append(referrer)
    return bool(pending)


@dataclass(frozen=True)
class _ClauseIndex:
    """Where each clause of the template's CNF came from.

    Lets the solver be handed only the clauses that can decide whether a
    set of pinned conditions is consistent (see Conditions._relevant_cnf).
    Clauses are the encoded ``set[int]`` of the CNF they were taken from.
    """

    # condition name -> clauses of ``Symbol(name) <-> expression``
    definitions: dict[str, list[set[int]]]
    # condition name -> names of the conditions its expression uses
    references: dict[str, set[str]]
    # condition name -> hashes of the Fn::Equals its expression uses
    equals: dict[str, set[str]]
    # Fn::Equals hash -> hash of the reference it compares
    reference_of: dict[str, str]
    # reference hash -> clauses saying at most one of its values holds
    exclusive: dict[str, list[set[int]]]
    # the parameter AllowedValues clauses
    allowed_values: list[set[int]]


@dataclass(frozen=True)
class Conditions:
    conditions: dict[str, Condition] = field(init=True, default_factory=dict)
    # The template's clauses: condition definitions, Fn::Equals exclusivity
    # and parameter AllowedValues. Never changed after creation and shared
    # by every Conditions evolved from this one; see ``cnf`` and
    # ``hard_cnf`` for the clauses that include the pinned conditions.
    _cnf: EncodedCNF = field(
        init=True, default_factory=EncodedCNF, compare=False, repr=False
    )
    _condition_symbols: dict[str, Symbol] = field(
        init=True, default_factory=dict, compare=False, repr=False
    )
    _index: _ClauseIndex | None = field(
        init=True, default=None, compare=False, repr=False
    )
    # Every condition pinned so far, and the subset pinned as facts (a
    # resource/output Condition, an enclosing Fn::If, a parameter or
    # condition definition) rather than *assumed* by condition-scenario
    # enumeration, which explores both values of a condition.
    _pins: Pins = field(init=True, default=frozenset(), compare=False, repr=False)
    _facts: Pins = field(init=True, default=frozenset(), compare=False, repr=False)
    # The Conditions this one was evolved from, or None for the template's
    # own. Evolving and reachability are pure functions of the template and
    # the pins, so their answers are kept once, on the root.
    _root: "Conditions | None" = field(
        init=True, default=None, compare=False, repr=False
    )
    _max_scenarios: int = field(init=False, default=128)
    # pins -> whether the template's clauses allow them
    _satisfiable: dict[Pins, bool] = field(
        init=False, default_factory=dict, compare=False, repr=False
    )
    # (pins, facts) -> the evolved Conditions
    _evolved: dict[tuple[Pins, Pins], "Conditions"] = field(
        init=False, default_factory=dict, compare=False, repr=False
    )
    # (name, value) -> the Condition pinned to that value
    _pinned: dict[tuple[str, bool], Condition] = field(
        init=False, default_factory=dict, compare=False, repr=False
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
        for k, v in conditions.items():
            if not isinstance(k, str):
                # Condition logical IDs must be strings (rule E8001 reports
                # non-string names).  Skip them here so the sympy Symbol
                # table below never receives a non-string name, which would
                # raise TypeError before any rule gets to run.
                continue
            try:
                other_conditions = conditions.copy()
                del other_conditions[k]
                obj[k] = Condition.create_from_instance(v, other_conditions)
            except ValueError:
                obj[k] = Condition.create_from_instance(
                    {"Fn::Equals": [None, None]}, conditions
                )

        # Build EncodedCNF with a Symbol per condition name
        cnf = EncodedCNF()
        condition_symbols: dict[str, Symbol] = {}
        definitions: dict[str, list[set[int]]] = {}
        references: dict[str, set[str]] = {}
        equals_of: dict[str, set[str]] = {}
        for name, cond in obj.items():
            sym = Symbol(name)
            condition_symbols[name] = sym
            # Add equivalence: Symbol(name) <-> condition's boolean expression
            start = len(cnf.data)
            add_equivalence(cnf, sym, cond.cnf)
            definitions[name] = cnf.data[start:]
            references[name] = {str(s) for s in cond.cnf.free_symbols if str(s) in obj}
            equals_of[name] = {e.hash for e in cond.equals}

        # Two Fn::Equals comparing the same single-valued reference (a
        # parameter, with or without AllowedValues, or a pseudo-parameter
        # such as AWS::Region or AWS::AccountId) to different static values
        # cannot both be true. Previously this constraint was only added
        # inside the AllowedValues loop below, so plain parameters and
        # pseudo-parameters never got it (issue #4799).
        equals_by_ref: dict[str, dict[str, list[Symbol]]] = {}
        reference_of: dict[str, str] = {}
        # A condition's equals include those of the conditions it references,
        # so one Fn::Equals can appear many times. Each is encoded only once.
        seen_equals: set[str] = set()
        for _, c_v in obj.items():
            for i in c_v.equals:
                if not isinstance(i.left.instance, str):
                    continue
                if not isinstance(i.right.instance, dict):
                    continue
                if i.hash in seen_equals:
                    continue
                seen_equals.add(i.hash)
                reference_of[i.hash] = i.right.hash
                equals_by_ref.setdefault(i.right.hash, {}).setdefault(
                    i.left.instance, []
                ).append(i.cnf)
        # Bitwise at-most-one: every static value of a reference gets a
        # distinct binary code over ceil(log2(n)) helper symbols, and each
        # Fn::Equals symbol implies the bits of its value's code. Two
        # Fn::Equals with different values force different bit patterns and so
        # cannot both be true. This needs n*log2(n) clauses instead of the
        # n*(n-1)/2 of a pairwise encoding.
        exclusive: dict[str, list[set[int]]] = {}
        for ref_hash, by_value in equals_by_ref.items():
            if len(by_value) < 2:
                continue
            start = len(cnf.data)
            width = (len(by_value) - 1).bit_length()
            bits = [Dummy("value_bit") for _ in range(width)]
            for code, value_symbols in enumerate(by_value.values()):
                for k, bit in enumerate(bits):
                    literal = bit if (code >> k) & 1 else Not(bit)
                    for value_symbol in value_symbols:
                        cnf.add_prop(Or(Not(value_symbol), literal))
            exclusive[ref_hash] = cnf.data[start:]

        # Add parameter AllowedValues constraints
        start = len(cnf.data)
        for p_k, p_v in parameters.items():
            if not p_v.allowed_values:
                continue
            allowed_values = p_v.allowed_values.copy()
            equals_cnfs: list[Symbol] = []
            for _, c_v in obj.items():
                for i in c_v.equals:
                    if i.right.hash == get_hash({"Ref": p_k}):
                        if not isinstance(i.left.instance, str):
                            continue
                        equals_cnfs.append(i.cnf)
                        if i.left.instance in allowed_values:
                            allowed_values.remove(i.left.instance)

            if not allowed_values and equals_cnfs:
                cnf.add_prop(Or(*equals_cnfs))

        # _relevant_cnf relies on the conditions forming a DAG. A condition
        # that refers to itself (directly or through others) is rejected by
        # CloudFormation, but a template can still contain one; such a
        # template gets no index and is solved with the full CNF.
        index: _ClauseIndex | None = None
        if not _has_cycle(references):
            index = _ClauseIndex(
                definitions=definitions,
                references=references,
                equals=equals_of,
                reference_of=reference_of,
                exclusive=exclusive,
                allowed_values=cnf.data[start:],
            )

        return cls(
            conditions=obj,
            _cnf=cnf,
            _condition_symbols=condition_symbols,
            _index=index,
        )

    @property
    def root(self) -> "Conditions":
        """The template's Conditions this one was evolved from."""
        return self if self._root is None else self._root

    def _pinned_cnf(self, pins: Pins) -> EncodedCNF:
        """Return a copy of the template's clauses with ``pins`` added."""
        cnf = copy_cnf(self.root._cnf)
        for name, value in pins:
            sym = self._condition_symbols.get(name)
            if sym is not None:
                cnf.add_prop(sym if value else Not(sym))
        return cnf

    @property
    def cnf(self) -> EncodedCNF:
        """The template's clauses plus every pinned condition."""
        return self._pinned_cnf(self._pins)

    @property
    def hard_cnf(self) -> EncodedCNF:
        """The template's clauses plus the conditions pinned as facts.

        hard_cnf mirrors cnf but excludes pins that were only *assumed* by
        condition-scenario enumeration (see evolve(assumed=True)).  It
        therefore represents the conditions that are actually forced at this
        point in the template (resource/output Condition, enclosing Fn::If,
        parameter and condition definitions).  Reachability questions (W1028)
        must be answered against hard_cnf so that a value merely being
        explored by enumeration is not mistaken for an impossible one.
        """
        return self._pinned_cnf(self._facts)

    def _relevant_cnf(self, pinned: dict[str, bool]) -> EncodedCNF | None:
        """Return the part of the template's clauses that decides whether
        ``pinned`` is consistent, with ``pinned`` added, or None when the
        whole CNF is needed.

        A condition symbol is defined by one equivalence and appears nowhere
        else, so a condition that is neither pinned nor used by a pinned one
        is free and its defining clauses can be left out. An Fn::Equals that
        no kept condition uses only occurs in the at-most-one clauses of its
        reference, and in the AllowedValues clauses, both of which the
        remaining values of that reference can always satisfy on their own
        (one of them true, the rest false). The at-most-one clauses are kept
        whole per reference, so a kept Fn::Equals is constrained exactly as
        in the full CNF.
        """
        index = self.root._index
        if index is None or not set(pinned) <= set(index.definitions):
            return None

        closure: set[str] = set()
        stack = list(pinned)
        while stack:
            name = stack.pop()
            if name in closure:
                continue
            closure.add(name)
            stack.extend(index.references[name])
        data: list[set[int]] = []
        refs: set[str] = set()
        for name in closure:
            data.extend(index.definitions[name])
            for equals_hash in index.equals[name]:
                ref_hash = index.reference_of.get(equals_hash)
                if ref_hash is not None:
                    refs.add(ref_hash)
        for ref_hash in refs:
            data.extend(index.exclusive.get(ref_hash, ()))
        data.extend(index.allowed_values)
        encoding = self.root._cnf.encoding
        for name, value in pinned.items():
            number = encoding.get(self._condition_symbols[name])
            if number is None:
                # The condition's definition added no clause, so its symbol
                # was never encoded; let add_prop encode it.
                return None
            data.append({number if value else -number})

        # Re-number the variables so the solver only sees the ones that
        # occur in these clauses; it otherwise branches on every unused one.
        # Literal 0 is sympy's encoding of False and stays as it is.
        numbers: dict[int, int] = {}
        compact: list[set[int]] = []
        for clause in data:
            compact.append(
                {
                    (
                        0
                        if literal == 0
                        else (1 if literal > 0 else -1)
                        * numbers.setdefault(abs(literal), len(numbers) + 1)
                    )
                    for literal in clause
                }
            )
        symbols = self.root._cnf.symbols
        return EncodedCNF(
            compact, {symbols[old - 1]: new for old, new in numbers.items()}
        )

    def _pins_satisfiable(self, pins: Pins) -> bool:
        """Return whether the template's clauses allow ``pins``.

        The answer depends only on the template and the pins, so it is
        kept once per template.
        """
        cache = self.root._satisfiable
        if pins not in cache:
            pinned = dict(pins)
            cnf = self._relevant_cnf(pinned)
            if cnf is None:
                cnf = self._pinned_cnf(pins)
            cache[pins] = _is_satisfiable(cnf)
        return cache[pins]

    def evolve(self, status: dict[str, bool], assumed: bool = False) -> "Conditions":
        """Pin one or more conditions to a value.

        When ``assumed`` is True the pin is recorded only in ``cnf`` (used for
        pruning impossible scenarios) and not in ``hard_cnf``.  This is used by
        condition-scenario enumeration, which explores both values of a
        condition; those values are hypotheses, not facts, so a branch that
        merely disagrees with the scenario currently being enumerated must not
        be reported as unreachable.
        """
        if not status:
            return self

        all_same = True
        for condition, condition_status in status.items():
            if (
                condition in self.conditions
                and self.conditions[condition].status != condition_status
            ):
                all_same = False
                break
        if all_same:
            return self

        pins = set(self._pins)
        facts = set(self._facts)
        for condition, value in status.items():
            current = self.conditions.get(condition)
            if current is None:
                continue
            if current.status is not None and current.status != value:
                raise Unsatisfiable(new_status=status, current_status=self.status)
            if value is None:
                # not a pin: the condition stays unknown
                continue
            pins.add((condition, value))
            # Only the conditions being pinned by *this* call, and only when
            # they are facts (not enumeration hypotheses), are facts of the
            # result. Conditions pinned earlier stay facts or assumptions.
            if not assumed:
                facts.add((condition, value))

        if not self._pins_satisfiable(frozenset(pins)):
            raise Unsatisfiable(new_status=status, current_status=self.status)
        return self._evolved_to(frozenset(pins), frozenset(facts))

    def _evolved_to(self, pins: Pins, facts: Pins) -> "Conditions":
        root = self.root
        key = (pins, facts)
        if key not in root._evolved:
            conditions = dict(root.conditions)
            for pin in pins:
                if pin not in root._pinned:
                    root._pinned[pin] = root.conditions[pin[0]].evolve(status=pin[1])
                conditions[pin[0]] = root._pinned[pin]
            root._evolved[key] = self.__class__(
                conditions=conditions,
                _cnf=root._cnf,
                _condition_symbols=root._condition_symbols,
                _index=root._index,
                _pins=pins,
                _facts=facts,
                _root=root,
            )
        return root._evolved[key]

    def is_reachable(self, name: str, value: bool) -> bool:
        """Return whether ``name`` can take ``value`` given the forced facts.

        This is answered against ``hard_cnf`` so that a condition merely being
        explored by scenario enumeration is not treated as impossible.  An
        unknown condition is conservatively considered reachable.
        """
        if name not in self._condition_symbols:
            return True
        if (name, not value) in self._facts:
            return False
        return self._pins_satisfiable(self._facts | {(name, value)})

    def _build_conditions(self, conditions: set[str]) -> Iterator["Conditions"]:
        scenarios_attempted = 0
        for product in itertools.product([True, False], repeat=len(conditions)):
            params = dict(zip(conditions, product))
            try:
                yield self.evolve(params, assumed=True)
            except Unsatisfiable:
                pass

            scenarios_attempted += 1
            # On occassions people will use a lot of non-related conditions
            # this is fail safe to limit the maximum number of responses
            if scenarios_attempted >= self._max_scenarios:
                return

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
        obj = {}
        for name, c in self.conditions.items():
            if c.status is not None:
                obj[name] = c.status

        return obj
