"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

from __future__ import annotations

from itertools import product
from typing import Any, Callable, Iterable, Iterator

from cfnlint.conditions._boolean import ClauseSet
from cfnlint.conditions._cdcl import CdclSolver


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


class PinSolver:
    """Answers "is ``cnf`` satisfiable with these literals pinned".

    Every question either condition engine asks the solver is the template's
    CNF plus a set of unit clauses (a condition or an Fn::Equals pinned true
    or false).  Each distinct set of pins is solved once and the answer kept,
    so repeating a question (the same scenario for the next resource, the
    same region for the next condition) costs a dictionary lookup.

    One :class:`CdclSolver` holds the CNF for the life of the lint and takes
    each pin set as assumptions, so a question costs the propagation the
    pins cause rather than a read of the whole CNF.
    """

    def __init__(self, cnf: ClauseSet) -> None:
        self._cnf = cnf
        self._answers: dict[frozenset[tuple[int, bool]], bool] = {}
        self._models: dict[Any, tuple[list[tuple[bool, ...]], bool]] = {}
        self._solver: CdclSolver | None = None

    def _cdcl(self) -> CdclSolver:
        if self._solver is None:
            self._solver = CdclSolver(self._cnf.data, len(self._cnf.encoding))
        return self._solver

    def satisfiable(self, pins: Iterable[tuple[Any, bool]]) -> bool:
        """``pins`` pairs a Symbol of ``cnf`` with the value it is pinned to.
        A symbol pinned both ways is unsatisfiable; a symbol that occurs in
        no clause is free and constrains nothing."""
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
        return self._cdcl().solve(sorted(v if value else -v for v, value in pins))

    def models(
        self,
        pins: Iterable[tuple[Any, bool]],
        free: list[Any],
        limit: int,
    ) -> tuple[list[tuple[bool, ...]], bool]:
        """The distinct assignments of ``free`` (Symbols) that extend to a
        model of the CNF with ``pins`` held, in a deterministic order, and
        whether more than ``limit`` exist (the list is then cut at ``limit``
        by :func:`alternate_ends`).  A symbol that occurs in no clause is
        unconstrained: as a pin it constrains nothing, in ``free`` it takes
        both values."""
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
        loose = [s for s in free if s not in encoded_free and s not in pinned]
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
        CNF stays satisfiable under the pins and the prefix as assumptions.
        Each step is one assumption solve, and a prefix is solved once
        however many times the walks reach it."""
        solver = self._cdcl()
        base = [v if value else -v for v, value in pins]
        cache: dict[tuple[int, ...], bool] = {}

        def feasible(assigned: list[int]) -> bool:
            assumptions = tuple(sorted(base + assigned))
            answer = cache.get(assumptions)
            if answer is None:
                answer = cache[assumptions] = solver.solve(assumptions)
            return answer

        if not feasible([]):
            return [], False
        n = len(free)

        def walk(first: bool) -> Iterator[tuple[bool, ...]]:
            assigned: list[int] = []
            current: list[bool] = []

            def descend(depth: int) -> Iterator[tuple[bool, ...]]:
                if depth == n:
                    yield tuple(current)
                    return
                variable = free[depth]
                for value in (first, not first):
                    assigned.append(variable if value else -variable)
                    current.append(value)
                    if feasible(assigned):
                        yield from descend(depth + 1)
                    current.pop()
                    assigned.pop()

            return descend(0)

        return alternate_ends(walk, limit)
