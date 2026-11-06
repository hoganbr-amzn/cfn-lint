"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0

Satisfiability over parameter values.

A condition is a function of the parameter values it reads, so "can these
pins hold together" is decided by searching for one assignment of values to
the references (parameters, pseudo-parameters, other single-valued
expressions) that makes every pinned symbol take its pinned value.  Each
reference takes one of a few candidate values: its AllowedValues if declared,
otherwise the literals it is compared against plus "some other value" (one
per reference in a Ref-vs-Ref class, so any equality pattern between the
references of the class is reachable).  This is exact for Fn::Equals between
references and literals, including Ref-vs-Ref and Ref-vs-itself.

The model is split into connected components (symbols linked by the
references they read).  A component is answered by searching its points;
one on a condition-reference cycle is answered by the SAT solver on the
template's clauses.  A point search is bounded by ``STEP_BUDGET`` node
evaluations: a component whose search runs out is answered by SAT for the
rest of the lint, as is one that has spent ``COMPONENT_BUDGET`` evaluations
on searches in all, and after ``MAX_TRIPS`` searches ran out every component
is.  Every answer is exact for the model that gave it, and the budgets bound
the lint rather than one question.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Iterable, Iterator

from cfnlint.conditions._boolean import ClauseSet, Dummy, Symbol
from cfnlint.conditions._cdcl import CdclSolver
from cfnlint.conditions._model import OTHER, Node, PointModel, compile_expr, symbols_of
from cfnlint.conditions._solver import alternate_ends

LOGGER = logging.getLogger(__name__)

# largest point product tabulated for scenario questions; larger components
# are enumerated by point searches on prefixes
ENUM_MAX = 4096
# node evaluations one point search may spend (about 0.05 s) before SAT
# answers; the largest search seen on real templates is a few thousand
STEP_BUDGET = 200_000
# after this many searches ran out of budget, the rest of the lint is SAT
MAX_TRIPS = 10
# node evaluations one component may spend on searches in all (about 0.5 s)
# before its questions go to SAT: many searches each just under the budget
# cost as much as trips
COMPONENT_BUDGET = 2_000_000
# definitions evaluated recursively along Condition: references before the
# evaluation switches to an explicit stack
EVAL_DEPTH = 64


class _Budget(Exception):
    """A point search spent its STEP_BUDGET."""


class _ModelComponent:
    """One connected component of the model: its symbols, references,
    Rules, route ("points" or "sat") and the answers given so far."""

    def __init__(self) -> None:
        self.symbols: set = set()
        self.refs: list = []
        self.constraints: list[Node] = []
        self.route = "points"
        self.points = 1
        self.answers: dict[frozenset, bool] = {}
        # truth table of a small component: point index -> bit, one column
        # (an int) per symbol asked so far; built on first use
        self.columns: dict[Any, int] | None = None
        self.table_refs: list = []
        self.all_points = 0
        self.rules_mask = 0
        self.searched = 0  # node evaluations spent in searches, in all


class PointSolver:
    """Answers ``satisfiable(pins)`` and ``models(pins, free, limit)`` on the
    model's points where a component allows it and on the CNF otherwise.
    Each distinct question is solved once and the answer kept, so repeating
    it (the same scenario for the next resource) costs a dictionary lookup."""

    def __init__(self, cnf: ClauseSet, model: PointModel) -> None:
        self._cdcl: CdclSolver | None = None  # built on the first SAT question
        self.fallbacks = 0  # point searches that ran out of budget
        self._models: dict[Any, tuple[list[tuple[bool, ...]], bool]] = {}
        self._cnf = cnf
        self._model = model
        self._answers: dict[frozenset, bool] = {}
        self._built = False
        self._global: bool | None = None
        self._work = 0  # node evaluations so far, the unit of the budget
        self._depth = 0  # definitions being evaluated along Condition: refs
        self._cand: dict[Any, list] = {}  # reference -> candidate values
        # filled by _build on the first question
        self._nodes: dict[Any, Node] = {}  # definition symbol -> node
        self._def_reads: dict[Any, list] = {}  # definition -> definitions read
        self._deps: dict[Any, frozenset] = {}  # definition -> references read
        self._cyclic: set = set()  # definitions on a reference cycle
        self._free: set = set()  # symbols that are boolean references
        self._atom_terms: dict[Any, tuple] = {}  # atom -> its two terms
        self._comp_of: dict[Any, _ModelComponent] = {}
        self._components: list[_ModelComponent] = []
        self._static_constraints: list[Node] = []  # Rules reading no symbol

    # ----- construction
    def _build(self) -> None:
        self._built = True
        m = self._model
        unknown: set = set()
        for sym, expr in m.defs.items():
            self._nodes[sym] = compile_expr(expr, m, unknown)
        # the definitions each definition reads directly (Condition: refs)
        for sym, node in self._nodes.items():
            reads: set = set()
            symbols_of(node, reads)
            self._def_reads[sym] = [d for d in reads if d in self._nodes]
        cons = [compile_expr(e, m, unknown) for e in m.constraints]
        # every free symbol becomes its own boolean reference
        self._free = unknown
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
            symbols_of(node, s)
            for o in s:
                union(sym, o)
        for sym, (left, right) in m.atoms.items():
            find(sym)
            for t in (left, right):
                if t[0] == "R":
                    union(sym, ("ref", t[1]))
        for sym in self._free:
            find(sym)
        cons_root = []
        for node in cons:
            s = set()
            symbols_of(node, s)
            if s:
                first = min(s, key=repr)
                for o in s:
                    union(first, o)
                cons_root.append((first, node))
            else:
                cons_root.append((None, node))

        # components in a fixed order (their first symbol's name), so that
        # MAX_TRIPS, a count over the whole lint, routes the same components
        # to SAT whatever the hash seed
        comps: dict[Any, _ModelComponent] = {}
        for x in sorted(parent, key=repr):
            c = comps.setdefault(find(x), _ModelComponent())
            if isinstance(x, tuple) and x and x[0] == "ref":
                c.refs.append(x[1])
            else:
                c.symbols.add(x)
                self._comp_of[x] = c
                if x in self._free:
                    c.refs.append(("free", x))
        for first, node in cons_root:
            if first is None:
                self._static_constraints.append(node)
            else:
                comps[find(first)].constraints.append(node)
        self._components = list(comps.values())

        # dependencies of each def (refs read, through other defs), cycles
        self._def_deps()

        for c in self._components:
            self._candidates(c)
            self._route(c)
        # the symbols of the CNF the model says nothing about (all of them
        # for a CNF with no model) form one component answered by SAT
        unclaimed = [
            sym
            for sym in self._cnf.encoding
            if isinstance(sym, Symbol)
            and not isinstance(sym, Dummy)
            and sym not in self._comp_of
        ]
        if unclaimed:
            c = _ModelComponent()
            c.route = "sat"
            c.symbols = set(unclaimed)
            for sym in unclaimed:
                self._comp_of[sym] = c
            self._components.append(c)

    def _def_deps(self) -> None:
        """``_deps``: the references each definition reads, through the
        definitions it references; ``_cyclic``: the definitions on a
        condition-reference cycle.  Iterative, so a long Condition: chain
        does not exhaust the interpreter's recursion limit."""
        colour: dict[Any, int] = {}  # 1 on the current path, 2 done
        for root in self._nodes:
            if root in colour:
                continue
            path: list = []
            stack = [(root, False)]
            while stack:
                sym, ready = stack.pop()
                if ready:
                    path.pop()
                    out: set = set()
                    self._node_deps(self._nodes[sym], out)
                    self._deps[sym] = frozenset(out)
                    colour[sym] = 2
                    continue
                state = colour.get(sym, 0)
                if state == 2:
                    continue
                if state == 1:
                    self._cyclic.update(path[path.index(sym) :])
                    continue
                colour[sym] = 1
                path.append(sym)
                stack.append((sym, True))
                for other in self._def_reads[sym]:
                    if colour.get(other, 0) != 2:
                        stack.append((other, False))

    def _node_deps(self, node: Node, out: set) -> None:
        op = node[0]
        if op == "a":
            for t in self._atom_terms[node[1]]:
                if t[0] == "R":
                    out.add(t[1])
        elif op == "f":
            out.add(("free", node[1]))
        elif op == "d":
            # computed before any definition that reads it, except on a
            # cycle, whose component is routed to SAT
            out |= self._deps.get(node[1], frozenset())
        elif op == "not":
            self._node_deps(node[1], out)
        elif op in ("and", "or"):
            for child in node[1]:
                self._node_deps(child, out)

    def _item_deps(self, node: Node) -> frozenset:
        out: set = set()
        self._node_deps(node, out)
        return frozenset(out)

    def _candidates(self, c: _ModelComponent) -> None:
        """The candidate values of each reference of ``c`` and the
        component's point count, their product."""
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
            left, right = terms
            if left[0] == "R" and right[0] == "L":
                literals[left[1]].add(right[1])
            elif left[0] == "L" and right[0] == "R":
                literals[right[1]].add(left[1])
            elif left[0] == "R" and right[0] == "R" and left[1] != right[1]:
                a, b = rfind(left[1]), rfind(right[1])
                if a != b:
                    ref_parent[a] = b
        classes: dict[Any, list] = {}
        for r in c.refs:
            classes.setdefault(rfind(r), []).append(r)
        points = 1
        for members in classes.values():
            if len(members) == 1:
                r = members[0]
                cands: list[Any]
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
            # a Ref-vs-Ref class: every member may take any literal of the
            # class or one of as many "other" values as there are members
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

    def _route(self, c: _ModelComponent) -> None:
        """Route ``c`` to SAT when its points cannot be searched: a
        condition-reference cycle.  A component too large to search runs
        one search until STEP_BUDGET and goes to SAT then."""
        if any(s in self._cyclic for s in c.symbols):
            c.route = "sat"

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
            by_comp: dict[int, tuple[_ModelComponent, list]] = {}
            for sym, val in pinned.items():
                c = self._comp_of.get(sym)
                if c is None:
                    continue  # occurs nowhere: free
                by_comp.setdefault(id(c), (c, []))[1].append((sym, val))
            ans = all(self._solve(c, p) for c, p in by_comp.values())
        self._answers[key] = ans
        return ans

    def _solve_global(self) -> bool:
        for node in self._static_constraints:
            if self._eval(node, {}, {}) is False:
                return False
        return all(self._solve(c, []) for c in self._components)

    def _solve(self, c: _ModelComponent, pins: list) -> bool:
        key = frozenset(pins)
        ans = c.answers.get(key)
        if ans is not None:
            return ans
        if c.columns is not None:
            # a small component already tabulated for scenario enumeration
            # is answered from its table, exact and O(1) per pin, whatever
            # route the lint's budget has given it since
            mask = c.rules_mask
            for sym, value in pins:
                col = self._column(c, self._node_of(sym))
                mask &= col if value else c.all_points & ~col
            ans = mask != 0
        elif c.route == "sat" or (
            # the lint has spent its share of point searches that led
            # nowhere, or this component has: its questions go to SAT
            self.fallbacks >= MAX_TRIPS or c.searched > COMPONENT_BUDGET
        ):
            c.route = "sat"
            ans = self._sat(pins)
        else:
            try:
                ans = self._points(c, pins)
            except _Budget:
                # The budget bounds one search.  A component whose questions
                # need more than that is answered by SAT from now on, so a
                # template that asks many distinct hard questions about one
                # component pays the budget once, not once per question.
                self.fallbacks += 1
                c.route = "sat"
                LOGGER.debug(
                    "Point search over budget; %s references answered by SAT",
                    len(c.refs),
                )
                ans = self._sat(pins)
        c.answers[key] = ans
        return ans

    def _sat(self, pins: list) -> bool:
        """Exact answer under ``pins`` by the SAT solver, which holds the
        template's clauses for the life of the lint and answers each pin set
        by propagation from them.  A symbol in no clause constrains
        nothing."""
        encoding = self._cnf.encoding
        if self._cdcl is None:
            self._cdcl = CdclSolver(self._cnf.data, len(encoding))
        return self._cdcl.solve(
            sorted(encoding[s] if v else -encoding[s] for s, v in pins if s in encoding)
        )

    def _eval(self, node: Node, vals: dict, memo: dict):
        """Three-valued evaluation of ``node`` over the partial assignment
        ``vals`` (reference -> value): True, False, or None when a reference
        it needs has no value yet.  ``memo`` keeps the value of each
        definition reached.  Every call counts one node evaluation."""
        self._work += 1
        op = node[0]
        if op == "c":
            return node[1]
        if op == "a":
            left, right = self._atom_terms[node[1]]
            if left[0] == "L":
                a = left[1]
            else:
                a = vals.get(left[1])
                if a is None:
                    return None
            if right[0] == "L":
                b = right[1]
            else:
                b = vals.get(right[1])
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
            if self._depth >= EVAL_DEPTH:
                # a long Condition: chain: settle what this definition reads,
                # deepest first, without recursing along the chain
                self._eval_defs(sym, vals, memo)
            self._depth += 1
            v = self._eval(self._nodes[sym], vals, memo)
            self._depth -= 1
            memo[sym] = v
            return v
        if op == "not":
            v = self._eval(node[1], vals, memo)
            return None if v is None else not v
        if op == "and":
            res: bool | None = True
            for child in node[1]:
                v = self._eval(child, vals, memo)
                if v is False:
                    return False
                if v is None:
                    res = None
            return res
        if op == "or":
            res = False
            for child in node[1]:
                v = self._eval(child, vals, memo)
                if v is True:
                    return True
                if v is None:
                    res = None
            return res
        raise ValueError(op)

    def _eval_defs(self, root, vals: dict, memo: dict) -> None:
        """Memoise every definition ``root`` reads, directly or through
        other definitions, children first, by an explicit stack."""
        stack = [(d, False) for d in self._def_reads[root] if d not in memo]
        while stack:
            sym, ready = stack.pop()
            if ready:
                memo[sym] = self._eval(self._nodes[sym], vals, memo)
                continue
            if sym in memo:
                continue
            memo[sym] = None
            stack.append((sym, True))
            for other in self._def_reads[sym]:
                if other not in memo:
                    stack.append((other, False))

    def _points(self, c: _ModelComponent, pins: list) -> bool:
        """Search the points of ``c`` for one where every pin and every
        Rule holds; raises :class:`_Budget` past STEP_BUDGET evaluations."""
        items = []  # (node, expected)
        for sym, val in pins:
            items.append((self._node_of(sym), val))
        for node in c.constraints:
            items.append((node, True))
        if not items:
            return True
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
            left, right = self._atom_terms[node[1]]
            if left[0] == "R" and right[0] == "L":
                ref, literal = left[1], right[1]
            elif left[0] == "L" and right[0] == "R":
                ref, literal = right[1], left[1]
            else:
                continue
            if (
                literal not in self._cand[ref]
                or vals.setdefault(ref, literal) != literal
            ):
                return False
        deps = [self._item_deps(n) for n, _ in items]
        refs = sorted({r for d in deps for r in d if r not in vals}, key=repr)
        readers: dict[Any, list[int]] = {r: [] for r in refs}
        for i, d in enumerate(deps):
            for r in d:
                if r not in vals:
                    readers[r].append(i)
        status: list[bool | None] = [None] * len(items)
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

        try:
            return descend(0, undecided)
        finally:
            c.searched += self._work - start

    # ----- scenario enumeration
    def _node_of(self, sym) -> Node:
        if sym in self._nodes:
            return ("d", sym)
        if sym in self._atom_terms:
            return ("a", sym)
        return ("f", sym)

    def _column(self, c: _ModelComponent, node: Node) -> int:
        """Bitmask over the points of a small component of where ``node`` is
        true; the component's points are indexed in product order over its
        references.  One pass over the points per node, kept for the lint."""
        assert c.columns is not None
        key = node[1] if node[0] in ("a", "d", "f") else id(node)
        col = c.columns.get(key)
        if col is None:
            col = 0
            bit = 1
            refs = c.table_refs
            for point in itertools.product(*[self._cand[r] for r in refs]):
                if self._eval(node, dict(zip(refs, point)), {}):
                    col |= bit
                bit <<= 1
            c.columns[key] = col
        return col

    def _table_oracle(self, c: _ModelComponent, pins: list, free: list):
        """A prefix-feasibility test over the truth table of a small
        component (at most ENUM_MAX points): the points allowed by ``pins``
        and the component's Rules, split by each asked symbol in turn.  The
        table is shared by every question about the component, so a
        template with many distinct questions pays one pass per symbol."""
        if c.columns is None:
            c.columns = {}
            c.table_refs = sorted(c.refs, key=repr)
            c.all_points = (1 << c.points) - 1
            mask = c.all_points
            for node in c.constraints:
                mask &= self._column(c, node)
            c.rules_mask = mask
        mask = c.rules_mask
        for sym, value in pins:
            col = self._column(c, self._node_of(sym))
            mask &= col if value else c.all_points & ~col
        columns = [self._column(c, self._node_of(sym)) for sym in free]
        masks: dict[tuple, int] = {(): mask}

        def feasible(prefix: tuple) -> bool:
            m = masks.get(prefix)
            if m is None:
                parent = masks[prefix[:-1]]
                col = columns[len(prefix) - 1]
                m = parent & col if prefix[-1] else parent & ~col
                masks[prefix] = m
            return m != 0

        return feasible

    def models(
        self,
        pins: Iterable[tuple[Any, bool]],
        free: list[Any],
        limit: int,
    ) -> tuple[list[tuple[bool, ...]], bool]:
        """The satisfiable assignments of ``free`` (Symbols) under ``pins``
        (Symbol, value), at most ``limit`` of them
        taken from both ends of the True-first lexicographic order
        (``alternate_ends``), sorted True first, and whether more exist.  A
        point-routed component of at most ENUM_MAX points answers from its
        truth table; any other component is asked prefix by prefix, each
        prefix one cached ``_solve``."""
        pinned: dict[Any, bool] = {}
        for symbol, value in pins:
            if pinned.setdefault(symbol, value) != value:
                return [], False
        key = (frozenset(pinned.items()), tuple(free), limit)
        cached = self._models.get(key)
        if cached is None:
            cached = self._models[key] = self._enumerate(pinned, free, limit)
        return cached

    def _enumerate(self, pinned: dict, free: list, limit: int):
        if not self.satisfiable(pinned.items()):
            return [], False
        comp_pins: dict[int, list] = {}
        for sym, val in pinned.items():
            c = self._comp_of.get(sym)
            if c is not None:
                comp_pins.setdefault(id(c), []).append((sym, val))
        # per position: (fixed value or None, component or None)
        slots: list[tuple[bool | None, _ModelComponent | None]] = []
        comp_free: dict[int, list] = {}
        comps: dict[int, _ModelComponent] = {}
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
        # a point-routed component of at most ENUM_MAX points answers every
        # prefix from its truth table; any other is asked prefix by prefix
        oracle: dict[int, Any] = {}
        for cid, c in comps.items():
            oracle[cid] = (
                self._table_oracle(c, comp_pins.get(cid, []), comp_free[cid])
                if c.route == "points" and c.points <= ENUM_MAX
                else None
            )
        n = len(free)

        def feasible(c: _ModelComponent, assigned: dict[int, list]) -> bool:
            cid = id(c)
            test = oracle[cid]
            if test is not None:
                return bool(test(tuple(v for _, v in assigned[cid])))
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

        return alternate_ends(walk, limit)


def make_solver(cnf: ClauseSet, model: PointModel | None = None) -> PointSolver:
    """The solver for one engine's CNF; with no model every question is
    answered on the CNF."""
    return PointSolver(cnf, model if model is not None else PointModel())
