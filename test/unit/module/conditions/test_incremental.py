"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

import itertools

import pytest
from sympy import Symbol
from sympy.assumptions.cnf import EncodedCNF

from cfnlint.conditions import _incremental
from cfnlint.conditions._solver import PinSolver, alternate_ends

try:
    import pysat.solvers  # noqa: F401

    MODES = ["g", "pyinc", "pysat-minisat", "pysat-cadical"]
except ImportError:  # pragma: no cover - depends on the environment
    MODES = ["g", "pyinc"]


def _solver(clauses, nvars, mode, monkeypatch):
    """A PinSolver over ``clauses``: ``g`` is the copy-and-solve path through
    sympy, anything else an incremental backend named by CFNLINT_COND_SAT."""
    monkeypatch.setenv("CFNLINT_COND_SAT", mode)
    monkeypatch.setattr(_incremental, "_PYSAT_MIN_SIZE", 0)
    syms = [Symbol(f"x{i}") for i in range(nvars)]
    cnf = EncodedCNF()
    cnf.encoding = {s: i + 1 for i, s in enumerate(syms)}
    cnf.data = [set(c) for c in clauses]
    return PinSolver(cnf, incremental=mode != "g"), syms


def _true_first(models):
    return sorted(models, key=lambda m: tuple(not v for v in m))


def test_default_backend_is_the_pure_python_incremental_solver(monkeypatch):
    monkeypatch.delenv("CFNLINT_COND_SAT", raising=False)
    assert _incremental.sat_backend_name() == "pyinc"
    monkeypatch.setenv("CFNLINT_COND_SAT", "nonsense")
    assert _incremental.sat_backend_name() == "pyinc"


@pytest.mark.parametrize("mode", MODES)
def test_models_agree_with_brute_force(mode, monkeypatch):
    # three components: {1,2,3}, {4,5}, {6}; 6 is forced true
    clauses = [{1, 2}, {-2, 3}, {-1, -3}, {4, 5}, {-4, -5}, {6}]
    solver, s = _solver(clauses, 6, mode, monkeypatch)
    free = [s[0], s[4], s[2], s[5], s[1], s[3]]  # interleaves the components

    def holds(a):
        return all(any(a[abs(l)] == (l > 0) for l in c) for c in clauses)

    truth = _true_first(
        tuple(a[i + 1] for i in (0, 4, 2, 5, 1, 3))
        for bits in itertools.product((True, False), repeat=6)
        for a in [dict(zip(range(1, 7), bits))]
        if holds(a)
    )
    assert solver.models([], free, 128) == (truth, False)
    assert solver.models([(s[3], True)], free, 128) == (
        [m for m in truth if m[5] is True],
        False,
    )
    assert solver.models([(s[5], False)], free, 128) == ([], False)
    # a second enumeration on the same solver sees every model again
    assert solver.models([], free, 128) == (truth, False)
    assert solver.satisfiable([(s[0], True), (s[2], True)]) is False
    assert solver.satisfiable([(s[0], True), (s[4], True)]) is True


@pytest.mark.parametrize(
    "mode", ["pyinc", "pysat-minisat"] if len(MODES) > 2 else ["pyinc"]
)
def test_truncated_enumeration_keeps_both_ends(mode, monkeypatch):
    # 10 independent variables: 1,024 models, cut at 128
    clauses = [{i, -i, 11} for i in range(1, 11)] + [{11}]
    solver, s = _solver(clauses, 11, mode, monkeypatch)
    found, truncated = solver.models([], s[:10], 128)
    assert truncated is True
    assert len(found) == 128 == len(set(found))
    assert (True,) * 10 in found
    assert (False,) * 10 in found
    assert found == _true_first(found)
    # the kept set depends only on the order of the asked symbols: 64 from
    # each end of the True-first lexicographic order over ``free``
    everything = _true_first(itertools.product((True, False), repeat=10))
    assert set(found) == set(everything[:64]) | set(everything[-64:])
    # an exact count at the limit is not a truncation
    found, truncated = solver.models([], s[:7], 128)
    assert truncated is False
    assert len(found) == 128
    found, truncated = solver.models([], s[:7], 127)
    assert truncated is True
    assert len(found) == 127


def test_alternate_ends_stops_when_the_walks_meet():
    models = _true_first(itertools.product((True, False), repeat=4))
    pulled = []

    def walk(first):
        for model in models if first else reversed(models):
            pulled.append(model)
            yield model

    found, truncated = alternate_ends(walk, 128)
    assert found == models
    assert truncated is False
    # 16 models, then one duplicate pull from the end whose turn it was
    assert len(pulled) == 17
    pulled.clear()
    found, truncated = alternate_ends(walk, 6)
    assert truncated is True
    assert found == models[:3] + models[-3:]
    assert len(pulled) == 7
    assert alternate_ends(lambda first: iter(()), 4) == ([], False)


@pytest.mark.parametrize("mode", MODES)
def test_unencoded_and_pinned_free_symbols(mode, monkeypatch):
    clauses = [{1, 2}, {-1, -2}]
    solver, s = _solver(clauses, 2, mode, monkeypatch)
    loose = Symbol("loose")
    assert solver.models([], [s[0], loose], 128) == (
        [(True, True), (True, False), (False, True), (False, False)],
        False,
    )
    assert solver.models([(s[0], True)], [s[1], s[0]], 128) == (
        [(False, True)],
        False,
    )
    assert solver.models([(s[0], True), (s[0], False)], [s[1]], 128) == (
        [],
        False,
    )


@pytest.mark.parametrize("mode", MODES)
def test_free_variables_fixed_at_level_zero(mode, monkeypatch):
    # 1 is forced true, 2 forced false; 3 and 4 are free and tied to them
    clauses = [{1}, {-2}, {-1, 3, 4}, {2, -3, -4}]
    solver, s = _solver(clauses, 4, mode, monkeypatch)
    for free in ([s[0], s[2], s[3]], [s[2], s[0], s[1], s[3]], [s[1], s[0]]):
        found, truncated = solver.models([], free, 128)
        assert truncated is False
        assert len(found) == len(set(found))
        for model in found:
            values = dict(zip(free, model))
            assert values.get(s[0], True) is True
            assert values.get(s[1], False) is False
            if s[2] in values and s[3] in values:
                assert values[s[2]] != values[s[3]]
    assert solver.models([], [s[0], s[2], s[3]], 128)[0] == [
        (True, True, False),
        (True, False, True),
    ]


@pytest.mark.parametrize("mode", [m for m in MODES if m != "g"])
def test_backend_solves_many_questions_on_one_clause_set(mode, monkeypatch):
    monkeypatch.setenv("CFNLINT_COND_SAT", mode)
    monkeypatch.setattr(_incremental, "_PYSAT_MIN_SIZE", 0)
    # a chain 1 -> 2 -> ... -> 8 plus an at-most-one over 1 and 8
    clauses = [[-i, i + 1] for i in range(1, 8)] + [[-1, -8]]
    backend = _incremental.make_backend(clauses, 8)
    assert backend.solve(()) is True
    assert backend.solve([1]) is False
    assert backend.solve([-1]) is True
    assert backend.solve([8]) is True
    assert backend.solve([-8, 3]) is False
    assert backend.solve([-8, -3]) is True
    assert backend.solve([1]) is False
    # an empty clause makes the component unsatisfiable under any pins
    assert _incremental.make_backend([[1], []], 1).solve(()) is False
    assert _incremental.make_backend([[1], [-1]], 1).solve(()) is False
