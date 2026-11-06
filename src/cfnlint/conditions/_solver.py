"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

from __future__ import annotations

from typing import Callable, Iterator


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
