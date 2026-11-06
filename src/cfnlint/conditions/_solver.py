"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

from __future__ import annotations

from typing import Any, Iterable

from sympy.assumptions.cnf import EncodedCNF
from sympy.logic.inference import satisfiable


class PinSolver:
    """Answers "is ``cnf`` satisfiable with these literals pinned".

    Every question either condition engine asks the solver is the template's
    CNF plus a set of unit clauses (a condition or an Fn::Equals pinned true
    or false).  Each distinct set of pins is solved once and the answer kept,
    so repeating a question (the same scenario for the next resource, the
    same region for the next condition) costs a dictionary lookup.
    """

    def __init__(self, cnf: EncodedCNF) -> None:
        self._cnf = cnf
        self._answers: dict[frozenset[tuple[int, bool]], bool] = {}

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
        cnf = self._cnf.copy()
        for variable, value in pins:
            cnf.data.append({variable if value else -variable})
        return bool(satisfiable(cnf))
