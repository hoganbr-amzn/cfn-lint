"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0

Incremental SAT backends for :class:`cfnlint.conditions._solver.PinSolver`.

A backend holds one clause set (one connected component of the template's
CNF, variables numbered 1..n) for the life of the lint and answers
``solve(assumptions)`` for many different pin sets without re-reading the
clauses.  Two implementations:

* :class:`PySatBackend` wraps a python-sat solver (MiniSat22, Glucose 4,
  CaDiCaL 1.9.5) when ``pysat`` is importable.
* :class:`PureBackend` is a small CDCL solver (two watched literals, 1UIP
  learning, assumptions) in plain Python with no dependency.

The backend is selected with ``CFNLINT_COND_SAT``: ``pyinc`` (default, the
pure-Python solver), ``pysat-minisat``, ``pysat-glucose``, ``pysat-cadical``
or ``auto`` (each falls back to ``pyinc`` when pysat is not importable).
Components of at most 32 variables always use the pure-Python solver.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

try:  # optional dependency
    from pysat.solvers import Solver as _PySatSolver  # type: ignore
except Exception:  # pragma: no cover - depends on the environment
    _PySatSolver = None

_PYSAT_NAMES = {
    "pysat-minisat": "minisat22",
    "pysat-glucose": "glucose4",
    "pysat-cadical": "cadical195",
}


def sat_backend_name() -> str:
    """The SAT backend for incremental solving: ``pyinc`` (default) or an
    optional pysat solver named by ``CFNLINT_COND_SAT`` (falls back to
    ``pyinc`` when pysat is not importable)."""
    mode = os.environ.get("CFNLINT_COND_SAT", "pyinc").strip().lower() or "pyinc"
    if mode == "auto":
        mode = "pysat-cadical" if _PySatSolver is not None else "pyinc"
    if mode not in _PYSAT_NAMES or _PySatSolver is None:
        mode = "pyinc"
    return mode


def make_backend(clauses: Iterable[Iterable[int]], size: int):
    """A persistent incremental solver for one component."""
    mode = sat_backend_name()
    if mode in _PYSAT_NAMES and size > _PYSAT_MIN_SIZE:
        return PySatBackend(_PYSAT_NAMES[mode], clauses, size)
    return PureBackend(clauses, size)


# A compiled solver instance costs memory (MiniSat and Glucose reserve a few
# MB of clause arena each; a template with 1,600 two-variable components ran
# out of address space under an 8 GB limit).  Components this small are
# trivial for the pure-Python solver, so only larger ones get a pysat solver.
_PYSAT_MIN_SIZE = int(os.environ.get("CFNLINT_COND_PYSAT_MIN_SIZE", "32"))


def _clean(clauses: Iterable[Iterable[int]]) -> tuple[list[list[int]], bool]:
    """Drop sympy's ``0`` literal (False); report whether an empty clause,
    which makes the component unsatisfiable, was present."""
    out: list[list[int]] = []
    empty = False
    for clause in clauses:
        lits = [lit for lit in clause if lit != 0]
        if not lits:
            empty = True
        out.append(lits)
    return out, empty


class PySatBackend:
    def __init__(self, name: str, clauses: Iterable[Iterable[int]], size: int) -> None:
        cleaned, self._unsat = _clean(clauses)
        self._solver = None
        if not self._unsat:
            self._solver = _PySatSolver(name=name, bootstrap_with=cleaned)

    def solve(self, assumptions: Sequence[int]) -> bool:
        if self._unsat:
            return False
        return bool(self._solver.solve(assumptions=list(assumptions)))

    def __del__(self) -> None:  # pragma: no cover - resource cleanup
        if self._solver is not None:
            try:
                self._solver.delete()
            except Exception:
                pass


class PureBackend:
    """A minimal CDCL solver with persistent clause database and assumptions.

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
        self.level = [0] * (size + 1)
        self.reason: list[list[int] | None] = [None] * (size + 1)
        self.trail: list[int] = []
        self.trail_lim: list[int] = []
        self.qhead = 0
        self.activity = [0.0] * (size + 1)
        self.var_inc = 1.0
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
        self.value[v] = 1 if lit > 0 else -1
        self.level[v] = len(self.trail_lim)
        self.reason[v] = reason
        self.trail.append(lit)

    def _new_level(self) -> None:
        self.trail_lim.append(len(self.trail))

    def _backtrack(self, level: int) -> None:
        if len(self.trail_lim) <= level:
            return
        start = self.trail_lim[level]
        for lit in self.trail[start:]:
            v = abs(lit)
            self.value[v] = 0
            self.reason[v] = None
        del self.trail[start:]
        del self.trail_lim[level:]
        self.qhead = len(self.trail)

    def _propagate(self) -> list[int] | None:
        """Unit propagation.  Returns the conflicting clause or None."""
        watches = self.watches
        value = self.value
        while self.qhead < len(self.trail):
            lit = self.trail[self.qhead]
            self.qhead += 1
            false_lit = -lit
            wl = watches[self._idx(lit)]  # clauses watching -lit... see _watch
            # _watch registers clause under idx(-w) for watched literal w, so
            # the clauses in watches[idx(lit)] have -lit as a watched literal,
            # which has just become false.
            i = 0
            keep: list[list[int]] = []
            n = len(wl)
            while i < n:
                clause = wl[i]
                i += 1
                if clause[0] == false_lit:
                    clause[0], clause[1] = clause[1], clause[0]
                # now clause[1] == false_lit
                first = clause[0]
                fv = value[abs(first)]
                if (fv == 1 and first > 0) or (fv == -1 and first < 0):
                    keep.append(clause)
                    continue
                found = False
                for k in range(2, len(clause)):
                    other = clause[k]
                    ov = value[abs(other)]
                    if not ((ov == 1 and other < 0) or (ov == -1 and other > 0)):
                        clause[1], clause[k] = other, false_lit
                        watches[self._idx(-other)].append(clause)
                        found = True
                        break
                if found:
                    continue
                keep.append(clause)
                if fv == 0:
                    self._enqueue(first, clause)
                else:
                    # conflict: copy remaining watchers back
                    keep.extend(wl[i:])
                    wl[:] = keep
                    return clause
            wl[:] = keep
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
                    self.activity[v] += self.var_inc
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
            clause = self.reason[abs(p)]
            assert clause is not None
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
        self.var_inc *= 1.05
        if self.var_inc > 1e100:
            self.activity = [a * 1e-100 for a in self.activity]
            self.var_inc *= 1e-100
        return learnt, bt

    # --- search ----------------------------------------------------------

    def _pick(self) -> int:
        best = 0
        best_act = -1.0
        value = self.value
        activity = self.activity
        for v in range(1, self.n + 1):
            if value[v] == 0 and activity[v] > best_act:
                best, best_act = v, activity[v]
        return -best  # negative polarity first (most cfn-lint pins are false)

    def solve(self, assumptions: Sequence[int]) -> bool:
        if self._unsat:
            return False
        assumptions = [a for a in assumptions if abs(a) <= self.n]
        result = self._search(assumptions)
        self._backtrack(0)
        return result

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
