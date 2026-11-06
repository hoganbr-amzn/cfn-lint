"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0

Parameter-point satisfiability (study prototype).

A condition is a function of the parameter values it reads, so "can these
pins hold together" is decided by searching for one assignment of values to
the references (parameters, pseudo-parameters, other single-valued
expressions) that makes every pinned symbol take its pinned value.  Each
reference takes one of a few candidate values: its AllowedValues if declared,
otherwise the literals it is compared against plus "some other value"
(one per reference in a Ref-vs-Ref class, so any equality pattern between
the references of the class is reachable).  This is exact for Fn::Equals
between references and literals, including Ref-vs-Ref and Ref-vs-itself.

Engine selection: CFNLINT_COND_ENGINE =
  hybrid  (default) a component whose static point count (after folding
          static intrinsics) is at most CFNLINT_PP_MAX_POINTS (1,048,576) is
          answered by points, the rest by incremental SAT; a points search
          over the work budget (CFNLINT_PP_BUDGET) routes its component to
          SAT for the rest of the lint, and after CFNLINT_PP_MAX_TRIPS such
          searches every component goes to SAT, so every answer is exact for
          the model that gave it and the budget bounds the lint, not one
          question.
  pp      every component by points except condition-reference cycles,
          with the same budget and SAT fallback.
  pyinc   every question by incremental SAT (PinSolver with persistent
          pure-Python CDCL backends), no points.
  g       the base PinSolver (copy-and-solve through sympy per question).
The SAT side is cfnlint.conditions._incremental (pure Python by default).
"""

from __future__ import annotations

import atexit
import itertools
import json
import os
from typing import Any, Iterable, Iterator

from sympy import Symbol
from sympy.assumptions.cnf import EncodedCNF
from sympy.logic.boolalg import (
    And,
    BooleanFalse,
    BooleanTrue,
    Equivalent,
    Implies,
    Not,
    Or,
)

from cfnlint.conditions._solver import PinSolver, alternate_ends
from cfnlint.conditions._utils import cfn_string

ENGINE = os.environ.get("CFNLINT_COND_ENGINE", "hybrid").strip().lower() or "hybrid"
if ENGINE not in ("hybrid", "pp", "pyinc", "g", "dual"):
    ENGINE = "hybrid"
# largest point product enumerated directly by models(); larger ones are
# enumerated by satisfiability checks on prefixes
ENUM_MAX = int(os.environ.get("CFNLINT_PP_ENUM_MAX", "4096"))
MAX_POINTS = int(os.environ.get("CFNLINT_PP_MAX_POINTS", "1048576"))
# node evaluations one point search may spend (about 0.05 s) before SAT
# answers; the largest search on the study's 765 non-adversarial templates
# is 5,412
STEP_BUDGET = int(os.environ.get("CFNLINT_PP_BUDGET", "200000"))
# after this many searches ran out of budget, the rest of the lint is SAT
MAX_TRIPS = int(os.environ.get("CFNLINT_PP_MAX_TRIPS", "10"))
MAX_REFS = 400  # recursion depth guard

_STATS: dict[str, Any] = {
    "engine": ENGINE,
    "pp_queries": 0,
    "sat_queries": 0,
    "fallbacks": 0,
    "pp_steps": 0,
    "pp_max": 0,
    "components": [],
    "disagree": 0,
    "disagree_samples": [],
}


def _dump_stats() -> None:
    path = os.environ.get("CFNLINT_COND_STATS")
    if not path:
        return
    comps = _STATS["components"]
    import sys

    summary = dict(_STATS)
    summary["argv"] = sys.argv[2] if len(sys.argv) > 2 else ""
    summary["components"] = len(comps)
    summary["pp_components"] = sum(1 for c in comps if c["route"] == "pp")
    summary["sat_components"] = sum(1 for c in comps if c["route"] == "sat")
    summary["sat_reasons"] = {}
    for c in comps:
        if c["route"] == "sat":
            summary["sat_reasons"][c["why"]] = summary["sat_reasons"].get(c["why"], 0) + 1
    big = sorted(comps, key=lambda c: -c["symbols"])[:5]
    summary["largest"] = big
    with open(path, "a") as f:
        f.write(json.dumps(summary) + "\n")


atexit.register(_dump_stats)

OTHER = "\x00other"


class PointModel:
    """What a condition engine knows about its symbols, beyond the CNF."""

    def __init__(self) -> None:
        # atom symbol -> (term, term); term = ("L", str) | ("R", key)
        self.atoms: dict[Any, tuple[tuple[str, Any], tuple[str, Any]]] = {}
        # condition symbol -> sympy expression
        self.defs: dict[Any, Any] = {}
        self.constraints: list[Any] = []
        # reference key -> list of string values (AllowedValues)
        self.domains: dict[Any, list[str]] = {}
        # reference keys that are parameters or pseudo-parameters
        self.plain_refs: set[Any] = set()
        self.free_atoms: set[Any] = set()

    def add_atom(self, symbol: Any, left: tuple, right: tuple) -> None:
        if not isinstance(symbol, Symbol):
            return
        self.atoms.setdefault(symbol, (left, right))

    def add_free(self, symbol: Any) -> None:
        if isinstance(symbol, Symbol):
            self.free_atoms.add(symbol)


def fold(value: Any, mappings: Any) -> str | None:
    """The string value of an Fn::Equals operand that needs no parameter:
    a scalar, or Fn::Select / Fn::FindInMap / Fn::Join over such values.
    Scalars are written as CloudFormation compares them (``cfn_string``:
    a boolean is "true"/"false"), the same form both engines' CNFs use."""
    if isinstance(value, (str, int, float, bool)):
        return str(cfn_string(value))
    if not isinstance(value, dict) or len(value) != 1:
        return None
    k, a = next(iter(value.items()))
    try:
        if k == "Fn::Select" and isinstance(a, list) and len(a) == 2:
            i = fold(a[0], mappings)
            if i is not None and isinstance(a[1], list):
                return fold(a[1][int(i)], mappings)
        if k == "Fn::FindInMap" and isinstance(a, list) and len(a) == 3:
            m, k1, k2 = (fold(x, mappings) for x in a)
            if None not in (m, k1, k2) and isinstance(mappings, dict):
                v = mappings[m][k1][k2]
                if isinstance(v, (str, int, float, bool)):
                    return str(cfn_string(v))
        if k == "Fn::Join" and isinstance(a, list) and len(a) == 2:
            if isinstance(a[0], str) and isinstance(a[1], list):
                parts = [fold(x, mappings) for x in a[1]]
                if None not in parts:
                    return a[0].join(parts)
    except (KeyError, IndexError, ValueError, TypeError):
        return None
    return None


# compiled node: (op, ...) with op in c, a, d, and, or, not
def _compile(expr: Any, atoms, defs, free, unknown) -> tuple:
    if isinstance(expr, BooleanTrue) or expr is True:
        return ("c", True)
    if isinstance(expr, BooleanFalse) or expr is False:
        return ("c", False)
    if isinstance(expr, Symbol):
        if expr in defs:
            return ("d", expr)
        if expr in atoms:
            return ("a", expr)
        unknown.add(expr)
        return ("f", expr)
    if isinstance(expr, Not):
        return ("not", _compile(expr.args[0], atoms, defs, free, unknown))
    if isinstance(expr, And):
        return ("and", tuple(_compile(a, atoms, defs, free, unknown) for a in expr.args))
    if isinstance(expr, Or):
        return ("or", tuple(_compile(a, atoms, defs, free, unknown) for a in expr.args))
    if isinstance(expr, Implies):
        a, b = expr.args
        return (
            "or",
            (
                ("not", _compile(a, atoms, defs, free, unknown)),
                _compile(b, atoms, defs, free, unknown),
            ),
        )
    if isinstance(expr, Equivalent):
        parts = [_compile(a, atoms, defs, free, unknown) for a in expr.args]
        a, b = parts[0], parts[1]
        return ("and", (("or", (("not", a), b)), ("or", (a, ("not", b)))))
    raise ValueError(f"unsupported expression {type(expr)}")


def _symbols(node: tuple, out: set) -> None:
    op = node[0]
    if op in ("a", "d", "f"):
        out.add(node[1])
    elif op == "not":
        _symbols(node[1], out)
    elif op in ("and", "or"):
        for c in node[1]:
            _symbols(c, out)


class _Budget(Exception):
    pass


class _Component:
    def __init__(self) -> None:
        self.symbols: set = set()
        self.refs: list = []
        self.constraints: list[tuple] = []
        self.route = "pp"
        self.why = ""
        self.points = 1
        self.cnf_indices: set[int] = set()
        self.answers: dict[frozenset, bool] = {}


class PointSolver:
    """Same interface as PinSolver: ``satisfiable(pins)``."""

    def __init__(self, cnf: EncodedCNF, model: PointModel, mode: str) -> None:
        self._pin = PinSolver(cnf, incremental=True)
        self.truncations = 0
        self._models: dict[Any, tuple[list[tuple[bool, ...]], bool]] = {}
        self._cnf = cnf
        self._model = model
        self._mode = mode
        self._answers: dict[frozenset, bool] = {}
        self._built = False
        self._global: bool | None = None
        self._trips = 0  # point searches that ran out of budget
        self._work = 0  # node evaluations so far, the unit of the budget
        self._dual = PinSolver(cnf) if ENGINE == "dual" else None

    # ----- construction
    def _build(self) -> None:
        self._built = True
        m = self._model
        unknown: set = set()
        self._nodes: dict[Any, tuple] = {}
        for sym, expr in m.defs.items():
            self._nodes[sym] = _compile(expr, m.atoms, m.defs, m.free_atoms, unknown)
        self._size_defs()
        cons = [
            _compile(e, m.atoms, m.defs, m.free_atoms, unknown) for e in m.constraints
        ]
        # every free symbol becomes its own boolean reference
        self._free = set(m.free_atoms) | unknown
        self._atom_terms = dict(m.atoms)

        parent: dict[Any, Any] = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for sym, node in self._nodes.items():
            find(sym)
            s: set = set()
            _symbols(node, s)
            for o in s:
                union(sym, o)
        for sym, (l, r) in m.atoms.items():
            find(sym)
            for t in (l, r):
                if t[0] == "R":
                    union(sym, ("ref", t[1]))
        for sym in self._free:
            find(sym)
        cons_root = []
        for node in cons:
            s = set()
            _symbols(node, s)
            if s:
                first = next(iter(s))
                for o in s:
                    union(first, o)
                cons_root.append((first, node))
            else:
                cons_root.append((None, node))

        comps: dict[Any, _Component] = {}
        self._comp_of: dict[Any, _Component] = {}
        for x in list(parent):
            c = comps.setdefault(find(x), _Component())
            if isinstance(x, tuple) and x and x[0] == "ref":
                c.refs.append(x[1])
            else:
                c.symbols.add(x)
                self._comp_of[x] = c
                if x in self._free:
                    c.refs.append(("free", x))
        self._static_constraints = []
        for first, node in cons_root:
            if first is None:
                self._static_constraints.append(node)
            else:
                comps[find(first)].constraints.append(node)
        self._components = list(comps.values())

        # dependencies of each def (refs read, through other defs), cycles
        self._deps: dict[Any, frozenset] = {}
        self._cyclic: set = set()
        for sym in self._nodes:
            self._def_deps(sym, [])

        encoding = self._cnf.encoding
        self._pin._split() if self._pin._components is None else None
        position = self._pin._position
        claimed: set[int] = set()
        for c in self._components:
            for s in c.symbols:
                v = encoding.get(s)
                if v is not None and v in position:
                    c.cnf_indices.add(position[v][0])
            claimed |= c.cnf_indices
            self._candidates(c)
            self._route(c)
            _STATS["components"].append(
                {
                    "route": c.route,
                    "why": c.why,
                    "symbols": len(c.symbols),
                    "refs": len(c.refs),
                    "points": c.points if c.points < 10**18 else ">1e18",
                }
            )
        self._unclaimed = set(range(len(self._pin._components))) - claimed

    def _def_deps(self, sym, stack) -> frozenset:
        if sym in self._deps:
            return self._deps[sym]
        if sym in stack:
            for s in stack[stack.index(sym) :]:
                self._cyclic.add(s)
            return frozenset()
        stack.append(sym)
        out: set = set()
        self._node_deps(self._nodes[sym], out, stack)
        stack.pop()
        res = frozenset(out)
        if sym not in self._cyclic:
            self._deps[sym] = res
        return res

    def _node_deps(self, node, out, stack) -> None:
        op = node[0]
        if op == "a":
            for t in self._atom_terms[node[1]]:
                if t[0] == "R":
                    out.add(t[1])
        elif op == "f":
            out.add(("free", node[1]))
        elif op == "d":
            out |= self._def_deps(node[1], stack)
        elif op == "not":
            self._node_deps(node[1], out, stack)
        elif op in ("and", "or"):
            for ch in node[1]:
                self._node_deps(ch, out, stack)

    def _item_deps(self, node) -> frozenset:
        out: set = set()
        self._node_deps(node, out, [])
        return frozenset(out)

    def _candidates(self, c: _Component) -> None:
        m = self._model
        literals: dict[Any, set] = {r: set() for r in c.refs}
        ref_parent: dict[Any, Any] = {r: r for r in c.refs}

        def rfind(x):
            while ref_parent[x] != x:
                ref_parent[x] = ref_parent[ref_parent[x]]
                x = ref_parent[x]
            return x

        for sym in c.symbols:
            terms = self._atom_terms.get(sym)
            if not terms:
                continue
            l, r = terms
            if l[0] == "R" and r[0] == "L":
                literals[l[1]].add(r[1])
            elif l[0] == "L" and r[0] == "R":
                literals[r[1]].add(l[1])
            elif l[0] == "R" and r[0] == "R" and l[1] != r[1]:
                a, b = rfind(l[1]), rfind(r[1])
                if a != b:
                    ref_parent[a] = b
        classes: dict[Any, list] = {}
        for r in c.refs:
            classes.setdefault(rfind(r), []).append(r)
        self._cand = getattr(self, "_cand", {})
        points = 1
        for members in classes.values():
            if len(members) == 1:
                r = members[0]
                if isinstance(r, tuple) and r and r[0] == "free":
                    cands = [True, False]
                elif r in m.domains:
                    dom = list(dict.fromkeys(m.domains[r]))
                    used = [v for v in dom if v in literals[r]]
                    rest = [v for v in dom if v not in literals[r]]
                    cands = used + rest[:1]
                else:
                    cands = sorted(literals[r]) + [OTHER]
                self._cand[r] = cands
                points *= len(cands)
                continue
            pool: set = set()
            for r in members:
                pool |= literals[r]
                pool |= set(m.domains.get(r, ()))
            others = [f"{OTHER}{i}" for i in range(len(members))]
            for r in members:
                if r in m.domains:
                    cands = list(dict.fromkeys(m.domains[r]))
                else:
                    cands = sorted(pool) + others
                self._cand[r] = cands
                points *= len(cands)
        c.points = points

    def _route(self, c: _Component) -> None:
        if any(s in self._cyclic for s in c.symbols):
            c.route, c.why = "sat", "cycle"
        elif len(c.refs) > MAX_REFS:
            c.route, c.why = "sat", "refs"
        elif self._mode == "hybrid" and c.points > MAX_POINTS:
            # route by static point count only: references of any kind
            # (parameters, pseudo-parameters, unfolded intrinsics) are
            # single values to points, so no other test is needed
            c.route, c.why = "sat", "points"
        # free-atom-only components and everything else: points

    # ----- queries
    def satisfiable(self, pins: Iterable[tuple[Any, bool]]) -> bool:
        pinned: dict[Any, bool] = {}
        for symbol, value in pins:
            if pinned.setdefault(symbol, value) != value:
                return False
        if not self._built:
            self._build()
        key = frozenset(pinned.items())
        ans = self._answers.get(key)
        if ans is not None:
            return ans
        if self._global is None:
            self._global = self._solve_global()
        if not self._global:
            ans = False
        else:
            by_comp: dict[int, tuple[_Component, list]] = {}
            for sym, val in pinned.items():
                c = self._comp_of.get(sym)
                if c is None:
                    continue  # occurs nowhere: free
                by_comp.setdefault(id(c), (c, []))[1].append((sym, val))
            ans = all(self._solve(c, p) for c, p in by_comp.values())
        self._answers[key] = ans
        if self._dual is not None:
            other = self._dual.satisfiable(pinned.items())
            if other != ans:
                _STATS["disagree"] += 1
                if len(_STATS["disagree_samples"]) < 10:
                    _STATS["disagree_samples"].append(
                        {"pp": ans, "g": other, "pins": sorted(f"{k}={v}" for k, v in pinned.items())[:12]}
                    )
        return ans

    def _solve_global(self) -> bool:
        for node in self._static_constraints:
            if self._eval(node, {}, {}) is False:
                return False
        for c in self._components:
            if not self._solve(c, []):
                return False
        if self._unclaimed:
            return self._pin.solve_components(sorted(self._unclaimed), [])
        return True

    def _solve(self, c: _Component, pins: list) -> bool:
        key = frozenset(pins)
        ans = c.answers.get(key)
        if ans is not None:
            return ans
        if c.route == "pp" and self._trips >= MAX_TRIPS:
            # the lint has spent its share of point searches that led
            # nowhere; every component still on points is answered by SAT
            c.route, c.why = "sat", "trips"
        if c.route == "sat":
            _STATS["sat_queries"] += 1
            ans = self._sat(c.cnf_indices, pins)
        else:
            _STATS["pp_queries"] += 1
            try:
                ans = self._points(c, pins)
            except _Budget:
                # The budget bounds one search.  A component whose questions
                # need more than that is answered by SAT from now on, so a
                # template that asks many distinct hard questions about one
                # component pays the budget once, not once per question.
                _STATS["fallbacks"] += 1
                self._trips += 1
                c.route, c.why = "sat", "budget"
                ans = self._sat(c.cnf_indices, pins)
        c.answers[key] = ans
        return ans

    def _sat(self, indices, pins) -> bool:
        """Exact answer for one component (its CNF components ``indices``)
        under ``pins``, by the incremental SAT backends."""
        encoding = self._cnf.encoding
        int_pins = [(encoding[s], v) for s, v in pins if s in encoding]
        return self._pin.solve_components(sorted(indices), int_pins)

    # three-valued evaluation over a partial assignment
    def _eval(self, node, vals, memo):
        self._work += 1
        op = node[0]
        if op == "c":
            return node[1]
        if op == "a":
            l, r = self._atom_terms[node[1]]
            if l[0] == "L":
                a = l[1]
            else:
                a = vals.get(l[1])
                if a is None:
                    if r[0] == "R" and r[1] == l[1]:
                        return True
                    return None
            if r[0] == "L":
                b = r[1]
            else:
                b = vals.get(r[1])
                if b is None:
                    return None
            return a == b
        if op == "f":
            return vals.get(("free", node[1]))
        if op == "d":
            sym = node[1]
            if sym in memo:
                return memo[sym]
            memo[sym] = None  # cycle guard (cyclic components never get here)
            v = self._eval(self._nodes[sym], vals, memo)
            memo[sym] = v
            return v
        if op == "not":
            v = self._eval(node[1], vals, memo)
            return None if v is None else not v
        if op == "and":
            res = True
            for ch in node[1]:
                v = self._eval(ch, vals, memo)
                if v is False:
                    return False
                if v is None:
                    res = None
            return res
        if op == "or":
            res = False
            for ch in node[1]:
                v = self._eval(ch, vals, memo)
                if v is True:
                    return True
                if v is None:
                    res = None
            return res
        raise ValueError(op)

    def _points(self, c: _Component, pins: list) -> bool:
        items = []  # (node, expected)
        for sym, val in pins:
            if sym in self._nodes:
                items.append((("d", sym), val))
            elif sym in self._atom_terms:
                items.append((("a", sym), val))
            else:
                items.append((("f", sym), val))
        for node in c.constraints:
            items.append((node, True))
        if not items:
            return True
        # A reader's items are evaluated until one fails, so put the cheap
        # ones first: a wide Or is only evaluated at the values that pass
        # the atoms pinned next to it.
        items.sort(key=lambda item: self._node_size(item[0]))
        # A pinned-true "Ref == literal" fixes its reference outright, so the
        # search never walks that reference's candidates (a parameter with
        # a thousand AllowedValues is a thousand steps per question).
        vals: dict[Any, Any] = {}
        for node, exp in items:
            if exp is not True:
                continue
            if node[0] == "d":
                node = self._nodes[node[1]]
            if node[0] != "a":
                continue
            l, r = self._atom_terms[node[1]]
            if l[0] == "R" and r[0] == "L":
                ref, literal = l[1], r[1]
            elif l[0] == "L" and r[0] == "R":
                ref, literal = r[1], l[1]
            else:
                continue
            if literal not in self._cand[ref] or vals.setdefault(ref, literal) != literal:
                return False
        deps = [self._item_deps(n) for n, _ in items]
        refs = sorted({r for d in deps for r in d if r not in vals}, key=repr)
        count = {r: 0 for r in refs}
        for d in deps:
            for r in d:
                if r not in vals:
                    count[r] += 1
        refs.sort(key=lambda r: (len(self._cand[r]) > 1, -count[r]))
        readers: dict[Any, list[int]] = {r: [] for r in refs}
        for i, d in enumerate(deps):
            for r in d:
                if r not in vals:
                    readers[r].append(i)
        status = [None] * len(items)
        undecided = 0
        start = self._work
        limit = start + STEP_BUDGET
        for i, (node, exp) in enumerate(items):
            v = self._eval(node, vals, {})
            if v is None:
                undecided += 1
            elif v != exp:
                return False
            status[i] = v
        if undecided == 0 or not refs:
            return undecided == 0
        cand = self._cand
        n = len(refs)

        def descend(depth: int, undecided: int) -> bool:
            r = refs[depth]
            idxs = readers[r]
            for value in cand[r]:
                vals[r] = value
                changed = []
                ok = True
                und = undecided
                for i in idxs:
                    if status[i] is not None:
                        continue
                    node, exp = items[i]
                    v = self._eval(node, vals, {})
                    if v is None:
                        continue
                    if v != exp:
                        ok = False
                        break
                    status[i] = v
                    changed.append(i)
                    und -= 1
                # the budget is in node evaluations, so a step that walks a
                # wide condition costs what it really costs
                if self._work > limit:
                    _STATS["pp_steps"] += self._work - start
                    raise _Budget()
                if ok:
                    if und == 0:
                        for i in changed:
                            status[i] = None
                        del vals[r]
                        return True
                    if depth + 1 < n and descend(depth + 1, und):
                        for i in changed:
                            status[i] = None
                        del vals[r]
                        return True
                for i in changed:
                    status[i] = None
            vals.pop(r, None)
            return False

        res = descend(0, undecided)
        _STATS["pp_steps"] += self._work - start
        if self._work - start > _STATS["pp_max"]:
            _STATS["pp_max"] = self._work - start
        return res

    def _node_size(self, node) -> int:
        """Node evaluations one evaluation of ``node`` can cost at most: its
        own nodes plus those of each definition it reads (``_def_size``)."""
        op = node[0]
        if op == "d":
            return self._def_size.get(node[1], 1)
        if op == "not":
            return 1 + self._node_size(node[1])
        if op in ("and", "or"):
            return 1 + sum(self._node_size(ch) for ch in node[1])
        return 1

    def _size_defs(self) -> None:
        """``_def_size`` for every definition, children first, without
        recursing along Condition: references (a chain can be long)."""
        self._def_size = {}
        visiting: set = set()
        for root in self._nodes:
            stack = [(root, False)]
            while stack:
                sym, ready = stack.pop()
                if sym in self._def_size:
                    continue
                if ready:
                    visiting.discard(sym)
                    self._def_size[sym] = 1 + self._node_size(self._nodes[sym])
                    continue
                if sym in visiting:
                    continue  # a cycle: sized as read
                visiting.add(sym)
                stack.append((sym, True))
                reads: set = set()
                _symbols(self._nodes[sym], reads)
                for other in reads:
                    if other in self._nodes and other not in self._def_size:
                        stack.append((other, False))

    # ----- scenario enumeration (Z-review D4)
    def _node_of(self, sym) -> tuple:
        if sym in self._nodes:
            return ("d", sym)
        if sym in self._atom_terms:
            return ("a", sym)
        return ("f", sym)

    def _projection(self, c: _Component, pins: list, free: list):
        """The set of prefixes of the distinct assignments of ``free`` (all in
        component ``c``) over the points that satisfy ``pins`` and the
        component's Rules, or None when the point product over the
        references they read is above ENUM_MAX."""
        items = [(self._node_of(s), v) for s, v in pins]
        items += [(node, True) for node in c.constraints]
        nodes = [self._node_of(s) for s in free]
        deps: set = set()
        for node, _ in items:
            deps |= self._item_deps(node)
        for node in nodes:
            deps |= self._item_deps(node)
        refs = sorted(deps, key=repr)
        size = 1
        for r in refs:
            size *= len(self._cand[r])
            if size > ENUM_MAX:
                return None
        prefixes: set = {()}
        cand = [self._cand[r] for r in refs]
        for point in itertools.product(*cand):
            vals = dict(zip(refs, point))
            memo: dict = {}
            ok = True
            for node, exp in items:
                if self._eval(node, vals, memo) is not exp:
                    ok = False
                    break
            if not ok:
                continue
            values = tuple(self._eval(node, vals, memo) for node in nodes)
            if None in values:
                return None
            for j in range(1, len(values) + 1):
                prefixes.add(values[:j])
        return prefixes

    def models(
        self,
        pins: Iterable[tuple[Any, bool]],
        free: list[Any],
        limit: int,
    ) -> tuple[list[tuple[bool, ...]], bool]:
        """Same contract as PinSolver.models: the satisfiable assignments of
        ``free`` under ``pins``, at most ``limit`` of them taken from both
        ends of the True-first lexicographic order (``alternate_ends``),
        sorted True first, and whether more exist.  A point-routed component
        enumerates its points over the references the asked conditions,
        the pins and its Rules read; any other component is asked prefix by
        prefix.  Exact for the model and deterministic."""
        pinned: dict[Any, bool] = {}
        for symbol, value in pins:
            if pinned.setdefault(symbol, value) != value:
                return [], False
        key = (frozenset(pinned.items()), tuple(free), limit)
        cached = self._models.get(key)
        if cached is None:
            cached = self._models[key] = self._enumerate(pinned, free, limit)
            if cached[1]:
                self.truncations += 1
        return cached

    def _enumerate(self, pinned, free, limit):
        if not self.satisfiable(pinned.items()):
            return [], False
        comp_pins: dict[int, list] = {}
        for sym, val in pinned.items():
            c = self._comp_of.get(sym)
            if c is not None:
                comp_pins.setdefault(id(c), []).append((sym, val))
        # per position: (fixed value or None, component or None)
        slots = []
        comp_free: dict[int, list] = {}
        comps: dict[int, _Component] = {}
        for sym in free:
            c = self._comp_of.get(sym)
            if sym in pinned:
                slots.append((pinned[sym], None))
                continue
            if c is None:
                slots.append((None, None))
                continue
            comps[id(c)] = c
            comp_free.setdefault(id(c), []).append(sym)
            slots.append((None, c))
        oracle: dict[int, Any] = {}
        for cid, c in comps.items():
            oracle[cid] = (
                self._projection(c, comp_pins.get(cid, []), comp_free[cid])
                if c.route == "pp"
                else None
            )
        n = len(free)

        def feasible(c: _Component, assigned: dict[int, list]) -> bool:
            cid = id(c)
            prefixes = oracle[cid]
            if prefixes is not None:
                return tuple(v for _, v in assigned[cid]) in prefixes
            return self._solve(c, comp_pins.get(cid, []) + assigned[cid])

        def walk(first: bool) -> Iterator[tuple[bool, ...]]:
            assigned: dict[int, list] = {cid: [] for cid in comps}
            current: list[bool] = []

            def descend(depth: int) -> Iterator[tuple[bool, ...]]:
                if depth == n:
                    yield tuple(current)
                    return
                fixed, c = slots[depth]
                for value in (first, not first) if fixed is None else (fixed,):
                    current.append(value)
                    if c is not None:
                        assigned[id(c)].append((free[depth], value))
                    if c is None or feasible(c, assigned):
                        yield from descend(depth + 1)
                    if c is not None:
                        assigned[id(c)].pop()
                    current.pop()

            return descend(0)

        # past ``limit``, keep both ends of the order (see alternate_ends)
        return alternate_ends(walk, limit)


def make_solver(cnf: EncodedCNF, model: PointModel | None):
    """The solver for one engine's CNF, by CFNLINT_COND_ENGINE."""
    if ENGINE in ("pp", "hybrid", "dual") and model is not None:
        return PointSolver(cnf, model, "pp" if ENGINE == "dual" else ENGINE)
    return PinSolver(cnf, incremental=ENGINE != "g")
