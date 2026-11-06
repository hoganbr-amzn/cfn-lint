"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0

What a condition engine knows about its symbols beyond the CNF: which
Fn::Equals each atom stands for, which values each reference may take, and
the boolean definition of each condition.  The point solver
(:mod:`cfnlint.conditions._points`) searches this model for parameter values.
"""

from __future__ import annotations

from typing import Any

from cfnlint.conditions._boolean import And, BooleanFalse, BooleanTrue, Not, Or, Symbol
from cfnlint.conditions._utils import cfn_string

# the candidate value standing for "any value the template does not compare
# the reference against"
OTHER = "\x00other"

# a term of an Fn::Equals: a literal or a reference by hash
Term = tuple[str, Any]


class PointModel:
    """The atoms, definitions, constraints and domains of one engine's CNF."""

    def __init__(self) -> None:
        # atom symbol -> (term, term); term = ("L", literal) | ("R", key)
        self.atoms: dict[Any, tuple[Term, Term]] = {}
        # condition symbol -> its boolean expression
        self.defs: dict[Any, Any] = {}
        # Rules, as boolean expressions that must hold
        self.constraints: list[Any] = []
        # reference key -> AllowedValues as CloudFormation strings
        self.domains: dict[Any, list[str]] = {}

    def add_equal(
        self,
        symbol: Any,
        left: tuple[Any, str],
        right: tuple[Any, str],
        mappings: Any,
    ) -> None:
        """Record ``symbol`` as the Fn::Equals of ``left`` and ``right``,
        each an operand and the hash of the operand.  A string operand, or
        one that folds to a string, is a literal; any other is a reference
        by its hash.  A constant ``symbol`` (a static Fn::Equals) is not an
        atom."""
        if not isinstance(symbol, Symbol):
            return
        self.atoms.setdefault(symbol, (_term(*left, mappings), _term(*right, mappings)))

    def add_domain(self, key: Any, allowed_values: Any) -> None:
        """Record the AllowedValues of reference ``key`` when every entry
        is a scalar, as the strings CloudFormation compares."""
        if (
            isinstance(allowed_values, list)
            and allowed_values
            and all(isinstance(v, (str, int, float, bool)) for v in allowed_values)
        ):
            self.domains[key] = [str(cfn_string(v)) for v in allowed_values]


def _term(value: Any, key: str, mappings: Any) -> Term:
    if isinstance(value, str):
        return ("L", value)
    folded = fold(value, mappings)
    if folded is not None:
        return ("L", folded)
    return ("R", key)


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
            # an index is a decimal digit string, as CloudFormation reads it
            if i is not None and i.isascii() and i.isdigit() and isinstance(a[1], list):
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
                    return a[0].join(parts)  # type: ignore[arg-type]
    except (KeyError, IndexError, ValueError, TypeError):
        return None
    return None


# A compiled node is a tuple (op, ...):
#   ("c", bool)         a constant
#   ("a", symbol)       an atom of the model (an Fn::Equals)
#   ("d", symbol)       a definition (a named condition)
#   ("f", symbol)       a free symbol, a boolean reference of its own
#   ("not", node)       negation
#   ("and", nodes) / ("or", nodes)
Node = tuple


def compile_expr(expr: Any, model: PointModel, unknown: set) -> Node:
    """Compile a boolean expression over the model's symbols; a Symbol that
    is neither an atom nor a definition (an Fn::Equals the model cannot
    evaluate, such as one with a missing operand) is collected in
    ``unknown`` and compiled as free, taking both values independently."""
    if isinstance(expr, BooleanTrue) or expr is True:
        return ("c", True)
    if isinstance(expr, BooleanFalse) or expr is False:
        return ("c", False)
    if isinstance(expr, Symbol):
        if expr in model.defs:
            return ("d", expr)
        if expr in model.atoms:
            return ("a", expr)
        unknown.add(expr)
        return ("f", expr)
    if isinstance(expr, Not):
        return ("not", compile_expr(expr.args[0], model, unknown))
    if isinstance(expr, And):
        return ("and", tuple(compile_expr(a, model, unknown) for a in expr.args))
    if isinstance(expr, Or):
        return ("or", tuple(compile_expr(a, model, unknown) for a in expr.args))
    raise ValueError(f"unsupported expression {type(expr)}")


def symbols_of(node: Node, out: set) -> None:
    """Collect the symbols a compiled node reads into ``out``."""
    op = node[0]
    if op in ("a", "d", "f"):
        out.add(node[1])
    elif op == "not":
        symbols_of(node[1], out)
    elif op in ("and", "or"):
        for child in node[1]:
            symbols_of(child, out)
