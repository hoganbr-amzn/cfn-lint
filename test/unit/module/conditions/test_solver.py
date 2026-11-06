"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

import itertools
import random
from unittest import TestCase

from cfnlint.conditions._boolean import ClauseSet, Symbol
from cfnlint.conditions._cdcl import CdclSolver
from cfnlint.conditions._model import PointModel
from cfnlint.conditions._points import PointSolver
from cfnlint.conditions._solver import alternate_ends


def _solver(clauses, nvars):
    """A PointSolver with no model (every question on the CNF) over
    ``clauses`` with Symbols x0..x(nvars-1)."""
    syms = [Symbol(f"x{i}") for i in range(nvars)]
    cnf = ClauseSet([set(c) for c in clauses], {s: i + 1 for i, s in enumerate(syms)})
    return PointSolver(cnf, PointModel()), syms


def _true_first(models):
    return sorted(models, key=lambda m: tuple(not v for v in m))


def _holds(clauses, assignment):
    """Whether ``assignment`` (variable -> bool) satisfies every clause."""
    return all(
        any(assignment[abs(lit)] == (lit > 0) for lit in clause) for clause in clauses
    )


class TestCnfQuestions(TestCase):
    def test_models_agree_with_brute_force(self):
        # three components: {1,2,3}, {4,5}, {6}; 6 is forced true
        clauses = [{1, 2}, {-2, 3}, {-1, -3}, {4, 5}, {-4, -5}, {6}]
        solver, s = _solver(clauses, 6)
        free = [s[0], s[4], s[2], s[5], s[1], s[3]]  # interleaves the components

        truth = _true_first(
            tuple(a[i + 1] for i in (0, 4, 2, 5, 1, 3))
            for bits in itertools.product((True, False), repeat=6)
            for a in [dict(zip(range(1, 7), bits))]
            if _holds(clauses, a)
        )
        self.assertEqual(solver.models([], free, 128), (truth, False))
        self.assertEqual(
            solver.models([(s[3], True)], free, 128),
            ([m for m in truth if m[5] is True], False),
        )
        self.assertEqual(solver.models([(s[5], False)], free, 128), ([], False))
        # a second enumeration on the same solver sees every model again
        self.assertEqual(solver.models([], free, 128), (truth, False))
        self.assertFalse(solver.satisfiable([(s[0], True), (s[2], True)]))
        self.assertTrue(solver.satisfiable([(s[0], True), (s[4], True)]))

    def test_truncated_enumeration_keeps_both_ends(self):
        # 10 independent variables: 1,024 models, cut at 128
        clauses = [{i, -i, 11} for i in range(1, 11)] + [{11}]
        solver, s = _solver(clauses, 11)
        found, truncated = solver.models([], s[:10], 128)
        self.assertTrue(truncated)
        self.assertEqual(len(found), 128)
        self.assertEqual(len(set(found)), 128)
        self.assertIn((True,) * 10, found)
        self.assertIn((False,) * 10, found)
        self.assertEqual(found, _true_first(found))
        # the kept set depends only on the order of the asked symbols: 64 from
        # each end of the True-first lexicographic order over ``free``
        everything = _true_first(itertools.product((True, False), repeat=10))
        self.assertEqual(set(found), set(everything[:64]) | set(everything[-64:]))
        # an exact count at the limit is not a truncation
        found, truncated = solver.models([], s[:7], 128)
        self.assertFalse(truncated)
        self.assertEqual(len(found), 128)
        found, truncated = solver.models([], s[:7], 127)
        self.assertTrue(truncated)
        self.assertEqual(len(found), 127)

    def test_alternate_ends_stops_when_the_walks_meet(self):
        models = _true_first(itertools.product((True, False), repeat=4))
        pulled = []

        def walk(first):
            for model in models if first else reversed(models):
                pulled.append(model)
                yield model

        found, truncated = alternate_ends(walk, 128)
        self.assertEqual(found, models)
        self.assertFalse(truncated)
        # 16 models, then one duplicate pull from the end whose turn it was
        self.assertEqual(len(pulled), 17)
        pulled.clear()
        found, truncated = alternate_ends(walk, 6)
        self.assertTrue(truncated)
        self.assertEqual(found, models[:3] + models[-3:])
        self.assertEqual(len(pulled), 7)
        self.assertEqual(alternate_ends(lambda first: iter(()), 4), ([], False))

    def test_unencoded_and_pinned_free_symbols(self):
        clauses = [{1, 2}, {-1, -2}]
        solver, s = _solver(clauses, 2)
        loose = Symbol("loose")
        self.assertEqual(
            solver.models([], [s[0], loose], 128),
            ([(True, True), (True, False), (False, True), (False, False)], False),
        )
        self.assertEqual(
            solver.models([(s[0], True)], [s[1], s[0]], 128), ([(False, True)], False)
        )
        self.assertEqual(
            solver.models([(s[0], True), (s[0], False)], [s[1]], 128), ([], False)
        )

    def test_symbol_in_no_clause_constrains_nothing(self):
        # x2 has a variable number but occurs in no clause
        clauses = [{1, 2}, {-1, -2}]
        solver, s = _solver(clauses, 3)
        self.assertTrue(solver.satisfiable([(s[2], True)]))
        self.assertTrue(solver.satisfiable([(s[2], False), (s[0], True)]))
        self.assertFalse(
            solver.satisfiable([(s[2], False), (s[0], True), (s[1], True)])
        )
        self.assertEqual(
            solver.models([(s[2], True)], [s[0], s[2]], 128),
            ([(True, True), (False, True)], False),
        )
        found, truncated = solver.models([], [s[2], s[0]], 128)
        self.assertFalse(truncated)
        self.assertEqual(
            sorted(found), [(False, False), (False, True), (True, False), (True, True)]
        )

    def test_free_variables_fixed_at_level_zero(self):
        # 1 is forced true, 2 forced false; 3 and 4 are free and tied to them
        clauses = [{1}, {-2}, {-1, 3, 4}, {2, -3, -4}]
        solver, s = _solver(clauses, 4)
        for free in ([s[0], s[2], s[3]], [s[2], s[0], s[1], s[3]], [s[1], s[0]]):
            found, truncated = solver.models([], free, 128)
            self.assertFalse(truncated)
            self.assertEqual(len(found), len(set(found)))
            for model in found:
                values = dict(zip(free, model))
                self.assertTrue(values.get(s[0], True))
                self.assertFalse(values.get(s[1], False))
                if s[2] in values and s[3] in values:
                    self.assertNotEqual(values[s[2]], values[s[3]])
        self.assertEqual(
            solver.models([], [s[0], s[2], s[3]], 128)[0],
            [(True, True, False), (True, False, True)],
        )


class TestCdclSolver(TestCase):
    def test_answers_many_questions_on_one_clause_set(self):
        # a chain 1 -> 2 -> ... -> 8 plus an at-most-one over 1 and 8
        clauses = [[-i, i + 1] for i in range(1, 8)] + [[-1, -8]]
        solver = CdclSolver(clauses, 8)
        self.assertTrue(solver.solve(()))
        self.assertFalse(solver.solve([1]))
        self.assertTrue(solver.solve([-1]))
        self.assertTrue(solver.solve([8]))
        self.assertFalse(solver.solve([-8, 3]))
        self.assertTrue(solver.solve([-8, -3]))
        self.assertFalse(solver.solve([1]))
        # an empty clause makes the component unsatisfiable under any pins
        self.assertFalse(CdclSolver([[1], []], 1).solve(()))
        self.assertFalse(CdclSolver([[1], [-1]], 1).solve(()))

    def test_assumption_outside_the_variables_is_refused(self):
        solver = CdclSolver([[1, 2]], 2)
        with self.assertRaises(ValueError):
            solver.solve([3])
        with self.assertRaises(ValueError):
            solver.solve([0])
        self.assertTrue(solver.solve([-1]))

    def test_learnt_clause_backjumps_to_its_second_highest_level(self):
        # decisions -1, -2, -3 (first unassigned variable, negative first);
        # the third propagates 5 from the first clause and the second clause
        # conflicts, so the learnt clause has one literal per level
        solver = CdclSolver([[1, 2, 3, 5], [1, 2, 3, -5]], 5)
        self.assertTrue(solver.solve(()))
        # asserting literal first, then the literal of the level jumped to
        self.assertEqual(solver.learnts, [[3, 2, 1]])
        self.assertFalse(solver.solve([-1, -2, -3]))
        self.assertTrue(solver.solve([-1, -2]))

    def test_pigeonhole_is_refuted_by_a_level_zero_conflict(self):
        # three pigeons, two holes: variable 2 * p + h is pigeon p in hole h
        clauses = [[2 * p + 1, 2 * p + 2] for p in range(3)]
        for h in (1, 2):
            clauses += [
                [-(2 * p + h), -(2 * q + h)] for p in range(3) for q in range(p)
            ]
        solver = CdclSolver(clauses, 6)
        self.assertFalse(solver.solve(()))
        self.assertTrue(solver._unsat)
        self.assertFalse(solver.solve([1]))

    def test_pigeonhole_learns_clauses_of_several_literals(self):
        # four pigeons, three holes: variable 3 * p + h is pigeon p in hole h
        clauses = [[3 * p + 1, 3 * p + 2, 3 * p + 3] for p in range(4)]
        for h in (1, 2, 3):
            clauses += [
                [-(3 * p + h), -(3 * q + h)] for p in range(4) for q in range(p)
            ]
        solver = CdclSolver(clauses, 12)
        self.assertFalse(solver.solve(()))
        self.assertTrue(all(len(learnt) > 1 for learnt in solver.learnts))
        self.assertFalse(solver.solve([1]))
        # without the fourth pigeon the three others fit
        three = [c for c in clauses[:3] + clauses[4:] if all(abs(x) <= 9 for x in c)]
        solver = CdclSolver(three, 9)
        self.assertTrue(solver.solve([1, 5]))
        self.assertFalse(solver.solve([1, 4]))

    def test_agrees_with_brute_force_across_many_questions(self):
        """Random clause sets over a few variables, each asked a run of
        random assumption sets on one solver, against brute force; the
        learned clauses one question leaves behind must not change the
        answer to the next."""
        rng = random.Random(1)
        for _ in range(300):
            n = rng.randint(1, 7)
            clauses = []
            for _ in range(rng.randint(0, 3 * n)):
                width = rng.randint(1, 3)
                clauses.append(
                    [rng.choice((-1, 1)) * rng.randint(1, n) for _ in range(width)]
                )
            if rng.random() < 0.05:
                clauses.append([])
            solver = CdclSolver(clauses, n)
            points = [
                dict(zip(range(1, n + 1), bits))
                for bits in itertools.product((True, False), repeat=n)
            ]
            for _ in range(rng.randint(1, 12)):
                assumptions = [
                    rng.choice((-1, 1)) * v
                    for v in rng.sample(range(1, n + 1), rng.randint(0, n))
                ]
                expected = any(
                    _holds(clauses, a)
                    and all(a[abs(lit)] == (lit > 0) for lit in assumptions)
                    for a in points
                )
                self.assertEqual(
                    solver.solve(assumptions), expected, (clauses, assumptions)
                )
                self.assertEqual(len(solver.trail_lim), 0)
