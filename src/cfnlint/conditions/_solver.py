"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Iterable, Iterator

from sympy.assumptions.cnf import EncodedCNF
from sympy.logic.inference import satisfiable

from cfnlint.conditions._boolean import ClauseSet


def alternate_ends(
    walk: Callable[[bool], Iterator[tuple[bool, ...]]], limit: int
) -> tuple[list[tuple[bool, ...]], bool]:
    """The truncation policy of a scenario enumeration.

    ``walk(first)`` yields the satisfiable assignments in lexicographic order
    with ``first`` tried before ``not first`` at every position, so
    ``walk(True)`` starts at the most-True end of the order and
    ``walk(False)`` at the most-False end.  Up to ``limit`` assignments are
    taken alternately from the two ends, so a truncated enumeration keeps
    the scenarios with most conditions true and those with most false (the
    all-false scenario resolves every nested Fn::If to its last leaf), and
    the kept set depends only on the order of the asked symbols, not on the
    solver.  Returns the assignments sorted True-first and whether more
    exist.  When the two walks meet (one yields an assignment the other
    already gave, or runs out) every assignment has been seen, so the
    enumeration is complete and nothing else is solved."""
    ends = (walk(True), walk(False))
    found: list[tuple[bool, ...]] = []
    seen: set[tuple[bool, ...]] = set()
    turn = 0
    truncated = False
    while True:
        model = next(ends[turn], None)
        turn ^= 1
        if model is None or model in seen:
            break
        if len(found) >= limit:
            truncated = True
            break
        found.append(model)
        seen.add(model)
    found.sort(key=lambda m: tuple(not v for v in m))
    return found, truncated


@dataclass(frozen=True)
class _Component:
    """The clauses of one connected component of the CNF, with their
    variables renumbered 1..size so the solver only sees these variables."""

    clauses: list[set[int]]
    size: int


class PinSolver:
    """Answers "is ``cnf`` satisfiable with these literals pinned".

    Every question either condition engine asks the solver is the template's
    CNF plus a set of unit clauses (a condition or an Fn::Equals pinned true
    or false).  Each distinct set of pins is solved once and the answer kept,
    so repeating a question (the same scenario for the next resource, the
    same region for the next condition) costs a dictionary lookup.

    A question is solved on the connected components its pins touch rather
    than on the whole CNF.  Two clauses are connected when they share a
    variable, so a condition's definition, the Fn::Equals it uses, the other
    Fn::Equals on the same references and the AllowedValues clauses of those
    references are all in one component, and a condition about unrelated
    parameters is in another.  Components share no variable, so the CNF is
    satisfiable with the pins exactly when every component is: the untouched
    ones are satisfiable whenever the whole CNF is (checked once), and the
    touched ones are solved together with the pins.
    """

    def __init__(self, cnf: ClauseSet) -> None:
        self._cnf = cnf
        self._answers: dict[frozenset[tuple[int, bool]], bool] = {}
        self._models: dict[Any, tuple[list[tuple[bool, ...]], bool]] = {}
        self._components: list[_Component] | None = None
        # variable -> (component index, variable number inside the component)
        self._position: dict[int, tuple[int, int]] = {}

    def satisfiable(self, pins: Iterable[tuple[Any, bool]]) -> bool:
        """``pins`` pairs a sympy Symbol of ``cnf`` with the value it is
        pinned to.  A symbol pinned both ways is unsatisfiable; a symbol that
        occurs in no clause is free and constrains nothing."""
        pinned: dict[Any, bool] = {}
        for symbol, value in pins:
            if pinned.setdefault(symbol, value) != value:
                return False
        encoding = self._cnf.encoding
        key = frozenset(
            (encoding[symbol], value)
            for symbol, value in pinned.items()
            if symbol in encoding
        )
        answer = self._answers.get(key)
        if answer is None:
            answer = self._answers[key] = self._solve(key)
        return answer

    def _solve(self, pins: frozenset[tuple[int, bool]]) -> bool:
        if self._components is None:
            self._split()
        assert self._components is not None
        if not pins:
            return all(
                self._is_satisfiable(component.clauses, component.size)
                for component in self._components
            )
        # Every component is satisfiable on its own, or nothing is.
        if frozenset() not in self._answers:
            self._answers[frozenset()] = self._solve(frozenset())
        if not self._answers[frozenset()]:
            return False
        touched = sorted({self._position[variable][0] for variable, _ in pins})
        offset = {}
        clauses: list[set[int]] = []
        size = 0
        for index in touched:
            component = self._components[index]
            offset[index] = size
            if size == 0:
                clauses.extend(component.clauses)
            else:
                clauses.extend(
                    {
                        literal + size if literal > 0 else literal - size
                        for literal in clause
                    }
                    for clause in component.clauses
                )
            size += component.size
        for variable, value in pins:
            index, number = self._position[variable]
            number += offset[index]
            clauses.append({number if value else -number})
        return self._is_satisfiable(clauses, size)

    def models(
        self,
        pins: Iterable[tuple[Any, bool]],
        free: list[Any],
        limit: int,
    ) -> tuple[list[tuple[bool, ...]], bool]:
        """The distinct assignments of ``free`` (sympy Symbols) that extend to
        a model of the CNF with ``pins`` held, in a deterministic order, and
        whether more than ``limit`` exist (the list is then cut at ``limit``
        by :func:`alternate_ends`).  A symbol of ``free`` that occurs in no
        clause is unconstrained and takes both values."""
        pinned: dict[Any, bool] = {}
        for symbol, value in pins:
            if pinned.setdefault(symbol, value) != value:
                return [], False
        encoding = self._cnf.encoding
        key_pins = frozenset(
            (encoding[s], v) for s, v in pinned.items() if s in encoding
        )
        encoded_free = [s for s in free if s in encoding and s not in pinned]
        key = (key_pins, tuple(encoding[s] for s in encoded_free), limit)
        cached = self._models.get(key)
        if cached is None:
            cached = self._models[key] = self._enumerate(
                key_pins, [encoding[s] for s in encoded_free], limit
            )
        found, truncated = cached
        # Expand back to the order of ``free``: pinned symbols keep their pin,
        # unencoded ones take both values.
        loose = [s for s in free if s not in encoding and s not in pinned]
        out: list[tuple[bool, ...]] = []
        for assignment in found:
            values = dict(zip(encoded_free, assignment))
            values.update({s: v for s, v in pinned.items()})
            for extra in product((True, False), repeat=len(loose)):
                values.update(zip(loose, extra))
                out.append(tuple(values[s] for s in free))
                if len(out) >= limit:
                    return out, truncated or len(out) < len(found) * (1 << len(loose))
        return out, truncated

    def _enumerate(
        self, pins: frozenset[tuple[int, bool]], free: list[int], limit: int
    ) -> tuple[list[tuple[bool, ...]], bool]:
        """Depth-first over ``free`` in order, keeping a prefix only when the
        touched components stay satisfiable with the pins and the prefix as
        unit clauses, so only prefixes of satisfiable assignments are solved
        and the work is one solve per feasible prefix rather than 2^len(free)
        attempts.  Each prefix is solved once however many times the walks
        reach it."""
        if self._components is None:
            self._split()
        assert self._components is not None
        if frozenset() not in self._answers:
            self._answers[frozenset()] = self._solve(frozenset())
        if not self._answers[frozenset()]:
            return [], False
        variables = {v for v, _ in pins} | set(free)
        touched = sorted({self._position[v][0] for v in variables})
        offset: dict[int, int] = {}
        clauses: list[set[int]] = []
        size = 0
        for index in touched:
            component = self._components[index]
            offset[index] = size
            clauses.extend(
                {lit + size if lit > 0 else lit - size for lit in clause}
                for clause in component.clauses
            )
            size += component.size

        def local(variable: int) -> int:
            index, number = self._position[variable]
            return number + offset[index]

        for variable, value in sorted(pins):
            clauses.append({local(variable) if value else -local(variable)})
        numbers = [local(v) for v in free]
        n = len(free)
        cache: dict[tuple[bool, ...], bool] = {}

        def feasible(prefix: tuple[bool, ...]) -> bool:
            answer = cache.get(prefix)
            if answer is None:
                units = [
                    {number if value else -number}
                    for number, value in zip(numbers, prefix)
                ]
                answer = cache[prefix] = self._is_satisfiable(clauses + units, size)
            return answer

        if not feasible(()):
            return [], False

        def walk(first: bool) -> Iterator[tuple[bool, ...]]:
            def descend(prefix: tuple[bool, ...]) -> Iterator[tuple[bool, ...]]:
                if len(prefix) == n:
                    yield prefix
                    return
                for value in (first, not first):
                    if feasible(prefix + (value,)):
                        yield from descend(prefix + (value,))

            return descend(())

        return alternate_ends(walk, limit)

    @staticmethod
    def _is_satisfiable(clauses: list[set[int]], size: int) -> bool:
        encoded = EncodedCNF(clauses, {n: n for n in range(1, size + 1)})
        return bool(satisfiable(encoded))

    def _split(self) -> None:
        """Group the clauses into connected components (union-find over
        the variables) and renumber each component's variables from 1."""
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for clause in self._cnf.data:
            variables = [abs(literal) for literal in clause if literal != 0]
            for other in variables[1:]:
                a, b = find(variables[0]), find(other)
                if a != b:
                    parent[a] = b

        members: dict[int, int] = {}
        components: list[list[set[int]]] = []
        sizes: list[int] = []
        for clause in self._cnf.data:
            variables = [abs(literal) for literal in clause if literal != 0]
            if not variables:
                root = -1  # the empty clause, or {0}: false on its own
            else:
                root = find(variables[0])
            index = members.get(root)
            if index is None:
                index = members[root] = len(components)
                components.append([])
                sizes.append(0)
            renumbered = set()
            for literal in clause:
                if literal == 0:
                    renumbered.add(0)
                    continue
                variable = abs(literal)
                if variable not in self._position:
                    sizes[index] += 1
                    self._position[variable] = (index, sizes[index])
                number = self._position[variable][1]
                renumbered.add(number if literal > 0 else -number)
            components[index].append(renumbered)
        self._components = [
            _Component(clauses, size) for clauses, size in zip(components, sizes)
        ]
