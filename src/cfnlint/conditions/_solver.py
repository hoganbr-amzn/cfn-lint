"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable

from sympy.assumptions.cnf import EncodedCNF
from sympy.logic.inference import satisfiable

from cfnlint.conditions._incremental import make_backend


def _product(n: int):
    return product((True, False), repeat=n)


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

    def __init__(self, cnf: EncodedCNF, incremental: bool = False) -> None:
        self._cnf = cnf
        # one persistent incremental solver per component instead of a
        # copy-and-solve through sympy per question
        self._incremental = incremental
        self._backends: list[Any] | None = None
        self._answers: dict[frozenset[tuple[int, bool]], bool] = {}
        self._models: dict[Any, tuple[list[tuple[bool, ...]], bool]] = {}
        # how many enumerations hit their limit (observable truncation)
        self.truncations = 0
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
        if self._backends is not None:
            return self._solve_incremental(pins)
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
                    {literal + size if literal > 0 else literal - size for literal in clause}
                    for clause in component.clauses
                )
            size += component.size
        for variable, value in pins:
            index, number = self._position[variable]
            number += offset[index]
            clauses.append({number if value else -number})
        return self._is_satisfiable(clauses, size)

    def _solve_incremental(self, pins: frozenset[tuple[int, bool]]) -> bool:
        """Each component keeps one loaded solver; a question is answered by
        solving every touched component under its pins as assumptions."""
        assert self._backends is not None
        if not pins:
            return all(backend.solve(()) for backend in self._backends)
        if frozenset() not in self._answers:
            self._answers[frozenset()] = self._solve(frozenset())
        if not self._answers[frozenset()]:
            return False
        return self.solve_components((), pins)

    def solve_components(
        self, indices: Iterable[int], pins: Iterable[tuple[int, bool]]
    ) -> bool:
        """Whether the components ``indices`` and those the (encoded) pins
        touch are satisfiable together with the pins, without the check that
        the rest of the CNF is.  Used by the point solver for the components
        it routes to SAT."""
        if self._components is None:
            self._split()
        grouped: dict[int, list[int]] = {index: [] for index in indices}
        for variable, value in pins:
            index, number = self._position[variable]
            grouped.setdefault(index, []).append(number if value else -number)
        if self._backends is not None:
            return all(
                self._backends[index].solve(sorted(assumptions))
                for index, assumptions in sorted(grouped.items())
            )
        assert self._components is not None
        clauses: list[set[int]] = []
        size = 0
        for index, assumptions in sorted(grouped.items()):
            component = self._components[index]
            clauses.extend(
                {lit + size if lit > 0 else lit - size for lit in clause}
                for clause in component.clauses
            )
            for lit in assumptions:
                clauses.append({lit + size if lit > 0 else lit - size})
            size += component.size
        if not clauses:
            return True
        return self._is_satisfiable(clauses, size)

    def models(
        self,
        pins: Iterable[tuple[Any, bool]],
        free: list[Any],
        limit: int,
    ) -> tuple[list[tuple[bool, ...]], bool]:
        """The distinct assignments of ``free`` (sympy Symbols) that extend to
        a model of the CNF with ``pins`` held, in a deterministic order, and
        whether more than ``limit`` exist (the list is then cut at ``limit``).

        Each assignment is found by one solve; its negation over ``free`` is
        then added as a blocking clause, so only satisfiable assignments cost
        a solve and the work is (number of real scenarios + 1) solves rather
        than 2^len(free) attempts.  A symbol of ``free`` that occurs in no
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
            if cached[1]:
                self.truncations += 1
        found, truncated = cached
        # Expand back to the order of ``free``: pinned symbols keep their pin,
        # unencoded ones take both values.
        loose = [s for s in free if s not in encoding and s not in pinned]
        out: list[tuple[bool, ...]] = []
        for assignment in found:
            values = dict(zip(encoded_free, assignment))
            values.update({s: v for s, v in pinned.items()})
            for extra in _product(len(loose)):
                values.update(zip(loose, extra))
                out.append(tuple(values[s] for s in free))
                if len(out) >= limit:
                    return out, truncated or len(out) < len(found) * (1 << len(loose))
        return out, truncated

    def _enumerate(
        self, pins: frozenset[tuple[int, bool]], free: list[int], limit: int
    ) -> tuple[list[tuple[bool, ...]], bool]:
        if self._components is None:
            self._split()
        assert self._components is not None
        if frozenset() not in self._answers:
            self._answers[frozenset()] = self._solve(frozenset())
        if not self._answers[frozenset()]:
            return [], False
        if self._backends is not None:
            return self._enumerate_incremental(pins, free, limit)
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
        found: list[tuple[bool, ...]] = []

        def ordered(truncated: bool):
            # True-first lexicographic order, as itertools.product gave
            return sorted(found, key=lambda m: tuple(not v for v in m)), truncated

        while True:
            model = satisfiable(EncodedCNF(clauses, {n: n for n in range(1, size + 1)}))
            if not model:
                return ordered(False)
            if len(found) >= limit:
                return ordered(True)
            assignment = tuple(bool(model[n]) for n in numbers)
            found.append(assignment)
            if not numbers:
                return ordered(False)
            # block this assignment of the free variables
            clauses.append(
                {-n if value else n for n, value in zip(numbers, assignment)}
            )

    def _enumerate_incremental(
        self, pins: frozenset[tuple[int, bool]], free: list[int], limit: int
    ) -> tuple[list[tuple[bool, ...]], bool]:
        """Depth-first over ``free`` in order, True before False, keeping a
        prefix only when the components it touches stay satisfiable under
        the pins and the prefix as assumptions.  Yields the lexicographically
        first ``limit`` assignments (True first), exactly, and whether more
        exist.  Each step is one assumption solve on one component."""
        assert self._backends is not None
        backends = self._backends
        base: dict[int, list[int]] = {}
        for variable, value in pins:
            index, number = self._position[variable]
            base.setdefault(index, []).append(number if value else -number)
        where = [self._position[v] for v in free]
        assigned: dict[int, list[int]] = {}
        cache: dict[tuple[int, tuple[int, ...]], bool] = {}

        def feasible(index: int) -> bool:
            assumptions = tuple(sorted(base.get(index, []) + assigned.get(index, [])))
            key = (index, assumptions)
            answer = cache.get(key)
            if answer is None:
                answer = cache[key] = backends[index].solve(assumptions)
            return answer

        # the pins alone
        for index in base:
            if not feasible(index):
                return [], False
        found: list[tuple[bool, ...]] = []
        current: list[bool] = []
        n = len(free)

        def descend(depth: int) -> bool:
            # returns True to stop (limit + 1 reached)
            if depth == n:
                if len(found) >= limit:
                    return True
                found.append(tuple(current))
                return False
            index, number = where[depth]
            for value in (True, False):
                assigned.setdefault(index, []).append(number if value else -number)
                current.append(value)
                stop = feasible(index) and descend(depth + 1)
                current.pop()
                assigned[index].pop()
                if stop:
                    return True
            return False

        truncated = descend(0)
        return found, truncated

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
        if self._incremental:
            self._backends = [
                make_backend(component.clauses, component.size)
                for component in self._components
            ]
