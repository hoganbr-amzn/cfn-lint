"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0

A small CDCL SAT solver with assumptions, for the condition engines.

A :class:`CdclSolver` holds one clause set (the template's CNF, variables
numbered 1..n) for the life of the lint and answers
``solve(assumptions)`` for many different pin sets without re-reading the
clauses.  Learned clauses are kept between calls.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def _clean(clauses: Iterable[Iterable[int]]) -> tuple[list[list[int]], bool]:
    """Drop the ``0`` literal (False); report whether an empty clause, which
    makes the component unsatisfiable, was present."""
    out: list[list[int]] = []
    empty = False
    for clause in clauses:
        lits = [lit for lit in clause if lit != 0]
        if not lits:
            empty = True
        out.append(lits)
    return out, empty


class CdclSolver:
    """A CDCL solver with a persistent clause database and assumptions.

    Literals are ints (``v`` or ``-v``, ``1 <= v <= size``).  Internally a
    literal ``l`` is indexed as ``2*v + (l < 0)`` so watch lists are plain
    lists.  Learned clauses are kept between calls; the trail is undone to
    level 0 at the end of every call, so every ``solve`` starts from the
    unit-propagated root state.
    """

    def __init__(self, clauses: Iterable[Iterable[int]], size: int) -> None:
        self.n = size
        cleaned, self._unsat = _clean(clauses)
        n2 = 2 * size + 2
        self.watches: list[list[list[int]]] = [[] for _ in range(n2)]
        # value per variable: 0 unassigned, 1 true, -1 false
        self.value = [0] * (size + 1)
        # the same per literal index (see _idx), read in _propagate
        self.lval = [0] * n2
        self.level = [0] * (size + 1)
        self.reason: list[list[int] | None] = [None] * (size + 1)
        self.trail: list[int] = []
        self.trail_lim: list[int] = []
        self.qhead = 0
        self.clauses: list[list[int]] = []
        self.learnts: list[list[int]] = []
        self.seen = [False] * (size + 1)
        if self._unsat:
            return
        for clause in cleaned:
            if not self._add_clause(clause):
                self._unsat = True
                return
        if self._propagate() is not None:
            self._unsat = True

    # --- clause database -------------------------------------------------

    @staticmethod
    def _idx(lit: int) -> int:
        return 2 * lit if lit > 0 else -2 * lit + 1

    def _lit_value(self, lit: int) -> int:
        v = self.value[abs(lit)]
        return v if lit > 0 else -v

    def _add_clause(self, lits: list[int]) -> bool:
        """Add an original clause at level 0.  Returns False on an
        immediate conflict."""
        seen: set[int] = set()
        out: list[int] = []
        for lit in lits:
            if -lit in seen:
                return True  # tautology
            if lit in seen:
                continue
            val = self._lit_value(lit)
            if val == 1:
                return True  # already satisfied at level 0
            if val == -1:
                continue  # false at level 0: drop
            seen.add(lit)
            out.append(lit)
        if not out:
            return False
        if len(out) == 1:
            self._enqueue(out[0], None)
            return True
        self.clauses.append(out)
        self._watch(out)
        return True

    def _watch(self, clause: list[int]) -> None:
        self.watches[self._idx(-clause[0])].append(clause)
        self.watches[self._idx(-clause[1])].append(clause)

    # --- assignment ------------------------------------------------------

    def _enqueue(self, lit: int, reason: list[int] | None) -> None:
        v = abs(lit)
        if lit > 0:
            self.value[v] = 1
            self.lval[2 * v] = 1
            self.lval[2 * v + 1] = -1
        else:
            self.value[v] = -1
            self.lval[2 * v] = -1
            self.lval[2 * v + 1] = 1
        self.level[v] = len(self.trail_lim)
        self.reason[v] = reason
        self.trail.append(lit)

    def _new_level(self) -> None:
        self.trail_lim.append(len(self.trail))

    def _backtrack(self, level: int) -> None:
        if len(self.trail_lim) <= level:
            return
        start = self.trail_lim[level]
        lval = self.lval
        for lit in self.trail[start:]:
            v = abs(lit)
            self.value[v] = 0
            lval[2 * v] = 0
            lval[2 * v + 1] = 0
            self.reason[v] = None
        del self.trail[start:]
        del self.trail_lim[level:]
        self.qhead = len(self.trail)

    def _propagate(self) -> list[int] | None:
        """Unit propagation.  Returns the conflicting clause or None.

        ``lval`` holds the truth of every literal by index (1 true, -1
        false, 0 unassigned), so no sign or ``abs`` is taken per literal, and
        each watch list is compacted in place."""
        watches = self.watches
        lval = self.lval
        trail = self.trail
        while self.qhead < len(trail):
            lit = trail[self.qhead]
            self.qhead += 1
            false_lit = -lit
            # _watch registers a clause under idx(-w) for each watched
            # literal w, so the clauses in watches[idx(lit)] have -lit as a
            # watched literal, which has just become false.
            wl = watches[2 * lit if lit > 0 else -2 * lit + 1]
            i = j = 0
            n = len(wl)
            while i < n:
                clause = wl[i]
                i += 1
                if clause[0] == false_lit:
                    clause[0], clause[1] = clause[1], clause[0]
                # now clause[1] == false_lit
                first = clause[0]
                fv = lval[2 * first if first > 0 else -2 * first + 1]
                if fv == 1:
                    wl[j] = clause
                    j += 1
                    continue
                for k in range(2, len(clause)):
                    other = clause[k]
                    if lval[2 * other if other > 0 else -2 * other + 1] != -1:
                        clause[1], clause[k] = other, false_lit
                        watches[2 * other + 1 if other > 0 else -2 * other].append(
                            clause
                        )
                        break
                else:
                    wl[j] = clause
                    j += 1
                    if fv == 0:
                        self._enqueue(first, clause)
                    else:
                        # conflict: keep the remaining watchers
                        while i < n:
                            wl[j] = wl[i]
                            i += 1
                            j += 1
                        del wl[j:]
                        return clause
            del wl[j:]
        return None

    # --- conflict analysis -------------------------------------------------

    def _analyze(self, conflict: list[int]) -> tuple[list[int], int]:
        """1UIP learning.  Returns (learnt clause with asserting literal
        first, backjump level)."""
        seen = self.seen
        level = self.level
        current = len(self.trail_lim)
        learnt: list[int] = [0]
        counter = 0
        p: int | None = None
        clause = conflict
        index = len(self.trail) - 1
        touched: list[int] = []
        while True:
            for q in clause:
                if q == p:
                    continue
                v = abs(q)
                if not seen[v] and level[v] > 0:
                    seen[v] = True
                    touched.append(v)
                    if level[v] >= current:
                        counter += 1
                    else:
                        learnt.append(q)
            while not seen[abs(self.trail[index])]:
                index -= 1
            p = self.trail[index]
            index -= 1
            seen[abs(p)] = False
            counter -= 1
            if counter <= 0:
                break
            reason = self.reason[abs(p)]
            assert reason is not None
            clause = reason
        learnt[0] = -p
        for v in touched:
            seen[v] = False
        if len(learnt) == 1:
            bt = 0
        else:
            max_i = 1
            for k in range(2, len(learnt)):
                if level[abs(learnt[k])] > level[abs(learnt[max_i])]:
                    max_i = k
            learnt[1], learnt[max_i] = learnt[max_i], learnt[1]
            bt = level[abs(learnt[1])]
        return learnt, bt

    # --- search ----------------------------------------------------------

    def _pick(self) -> int:
        """The first unassigned variable, negative polarity first (most
        cfn-lint pins are false); 0 when every variable is assigned."""
        value = self.value
        for v in range(1, self.n + 1):
            if value[v] == 0:
                return -v
        return 0

    def solve(self, assumptions: Sequence[int]) -> bool:
        """Whether the clauses are satisfiable with each literal of
        ``assumptions`` true.  Learned clauses and level-0 units are kept
        for the next call; the trail is always back at level 0 after one."""
        if any(not 1 <= abs(a) <= self.n for a in assumptions):
            raise ValueError(f"assumption outside variables 1..{self.n}")
        if self._unsat:
            return False
        try:
            return self._search(assumptions)
        finally:
            self._backtrack(0)

    def _search(self, assumptions: Sequence[int]) -> bool:
        n_assume = len(assumptions)
        while True:
            conflict = self._propagate()
            if conflict is not None:
                if len(self.trail_lim) == 0:
                    self._unsat = True
                    return False
                learnt, bt = self._analyze(conflict)
                self._backtrack(bt)
                if len(learnt) == 1:
                    self._enqueue(learnt[0], None)
                else:
                    self.learnts.append(learnt)
                    self._watch(learnt)
                    self._enqueue(learnt[0], learnt)
                continue
            # decide: assumptions first, in order
            d = len(self.trail_lim)
            if d < n_assume:
                lit = assumptions[d]
                val = self._lit_value(lit)
                if val == -1:
                    return False  # assumption contradicted
                self._new_level()
                if val == 0:
                    self._enqueue(lit, None)
                continue
            lit = self._pick()
            if lit == 0:
                return True
            self._new_level()
            self._enqueue(lit, None)
