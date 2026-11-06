"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from sympy.assumptions.cnf import EncodedCNF
from sympy.logic.inference import satisfiable


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

    def __init__(self, cnf: EncodedCNF) -> None:
        self._cnf = cnf
        self._answers: dict[frozenset[tuple[int, bool]], bool] = {}
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
