"""
Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

import functools
import hashlib
import json
from typing import Any, Hashable

from sympy import And, Dummy, Equivalent, Not, Or
from sympy.assumptions.cnf import EncodedCNF
from sympy.logic.boolalg import Boolean, BooleanFunction, to_nnf


class ObjectEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if hasattr(o, "_value"):
            # pylint: disable=protected-access
            return o._value
        return o


# Convert object to a hashable representation
def _make_hashable(obj: Any) -> Any:  # type: ignore # Actual return is Hashable but mypy can't verify
    """Convert an object to a hashable representation"""
    if isinstance(obj, bool):
        # Tag booleans with their type: ``True == 1`` and ``False == 0`` in
        # Python, so an untagged bool shares the cache key, and so the hash,
        # of the integer. CloudFormation compares ``true`` and ``1`` as the
        # strings "true" and "1", which differ, so the two must hash apart.
        return (bool, obj)
    # Fast path for already hashable types
    if isinstance(obj, (str, int, float, type(None))):
        return obj

    if isinstance(obj, dict):
        # Sort items once and convert to tuple
        items = sorted(obj.items())
        return tuple((k, _make_hashable(v)) for k, v in items)
    elif isinstance(obj, list):
        # Convert list to tuple of hashable items
        return tuple(_make_hashable(v) for v in obj)
    elif hasattr(obj, "_value"):
        return _make_hashable(obj._value)
    return obj


# Cache the hash calculation
@functools.lru_cache(maxsize=16384)  # Increased from 4096
def _cached_hash(hashable_obj: Hashable) -> str:
    """Cached hash calculation for hashable objects"""
    # Use a faster string representation for simple types
    if isinstance(hashable_obj, (str, int, float, bool, type(None))):
        return hashlib.sha1(str(hashable_obj).encode("utf-8")).hexdigest()

    # For complex types, use json for consistent serialization
    try:
        return hashlib.sha1(str(hashable_obj).encode("utf-8")).hexdigest()
    except Exception:  # pylint: disable=broad-except
        # Fallback to json for objects that can't be directly stringified
        return hashlib.sha1(
            json.dumps(hashable_obj, sort_keys=True, cls=ObjectEncoder).encode("utf-8")
        ).hexdigest()


def get_hash(o: Any) -> str:
    """Return a hash of an object
    This version prioritizes consistency over performance by always
    converting the object to a hashable representation before hashing.
    This ensures that identical content always produces the same hash,
    regardless of object identity.
    """
    # For simple hashable types, use direct hashing (bools are tagged by
    # _make_hashable so they do not share the cache key of 1 and 0)
    if isinstance(o, (str, int, float, type(None))) and not isinstance(o, bool):
        return _cached_hash(o)

    # Convert to hashable and compute hash
    try:
        hashable = _make_hashable(o)
        return _cached_hash(hashable)
    except Exception:  # pylint: disable=broad-except
        # Fallback to original method if conversion fails
        return hashlib.sha1(
            json.dumps(o, sort_keys=True, cls=ObjectEncoder).encode("utf-8")
        ).hexdigest()


def add_equivalence(cnf: EncodedCNF, symbol: Boolean, expr: Boolean) -> None:
    """Add ``symbol <-> expr`` to ``cnf`` without exponential growth.

    Handing sympy ``Equivalent(symbol, expr)`` directly makes it distribute
    And over Or, so an Or of n Ands becomes 2^n clauses. Instead every And/Or
    subterm of ``expr`` gets an auxiliary variable defined by a full
    equivalence (Tseitin encoding), which keeps the clause count linear in the
    size of ``expr``. Each auxiliary variable is fully determined by the
    original variables, so the satisfiability of ``cnf`` plus any constraint
    over the original variables is unchanged.

    ``symbol`` itself serves as the variable of the outermost And/Or, so a
    condition that is a single And/Or of literals costs no auxiliary variable
    at all and encodes to the same clauses as ``Equivalent`` would.
    """
    literal = _tseitin(cnf, expr, {}, root=symbol)
    if literal is not symbol:
        cnf.add_prop(Equivalent(symbol, literal))


def add_constraint(cnf: EncodedCNF, expr: Boolean) -> None:
    """Add ``expr`` to ``cnf`` as a constraint without exponential growth.

    Like ``add_equivalence`` this gives every And/Or subterm an auxiliary
    variable, so an Or of n Ands costs a linear number of clauses rather than
    the 2^n of sympy's distributive conversion.
    """
    cnf.add_prop(_tseitin(cnf, expr, {}))


def add_at_most_one(cnf: EncodedCNF, literals: list[Any]) -> None:
    """Add "at most one of ``literals`` is true" to ``cnf``.

    Up to five literals get the pairwise clauses ``Not(a & b)``. Longer
    lists get the binary encoding: ``ceil(log2 n)`` auxiliary bits, and
    each literal implies its own bit pattern, so two literals cannot both
    be true. That is ``n * ceil(log2 n)`` clauses instead of ``n(n-1)/2``.
    The bits are determined by the literals, so satisfiability with any
    constraint over the original variables is unchanged.
    """
    if len(literals) <= 5:
        for i, a in enumerate(literals):
            for b in literals[i + 1 :]:
                cnf.add_prop(Not(And(a, b)))
        return
    width = (len(literals) - 1).bit_length()
    bits = [Dummy("at_most_one") for _ in range(width)]
    for code, literal in enumerate(literals):
        for k, bit in enumerate(bits):
            cnf.add_prop(Or(Not(literal), bit if (code >> k) & 1 else Not(bit)))


def _tseitin(
    cnf: EncodedCNF, expr: Any, cache: dict[Any, Any], root: Any = None
) -> Any:
    """Return a literal equivalent to ``expr``, adding defining clauses
    for any auxiliary variables to ``cnf``. When ``root`` is given and
    ``expr`` is an And/Or, ``root`` is used as its variable instead of a
    fresh one."""
    if isinstance(expr, Not):
        return Not(_tseitin(cnf, expr.args[0], cache))
    if isinstance(expr, (And, Or)):
        if expr in cache:
            return cache[expr]
        literals = [_tseitin(cnf, arg, cache) for arg in expr.args]
        aux = Dummy("cond") if root is None else root
        if isinstance(expr, And):
            # aux -> each literal; all literals -> aux
            for literal in literals:
                cnf.add_prop(Or(Not(aux), literal))
            cnf.add_prop(Or(aux, *[Not(literal) for literal in literals]))
        else:
            # aux -> some literal; each literal -> aux
            cnf.add_prop(Or(Not(aux), *literals))
            for literal in literals:
                cnf.add_prop(Or(aux, Not(literal)))
        cache[expr] = aux
        return aux
    if isinstance(expr, BooleanFunction):
        # Implies, Equivalent, Xor, ... rewritten in terms of And/Or/Not
        nnf = to_nnf(expr, simplify=False)
        if nnf != expr:
            return _tseitin(cnf, nnf, cache, root)
    return expr
