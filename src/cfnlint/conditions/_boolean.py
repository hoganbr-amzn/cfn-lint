"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0

Boolean expressions and a clause set for the condition engines.

* ``Symbol(name)`` is equal to every other ``Symbol`` with the same name,
  ``Dummy`` only to itself; ``repr`` gives ``name`` and ``_name``, and the
  engines order symbols by it.
* ``And``/``Or`` flatten nested operators of the same kind, drop the
  identity, short-circuit on the absorbing constant, drop duplicate
  arguments and compare equal regardless of argument order; with no
  argument they are the identity and with one they are that argument.
  ``Not`` removes double negation and negates constants.  Complementary
  arguments (``And(x, Not(x))``) are kept.
* Anything that is not an expression (``None``, a string) raises
  ``ValueError``.
* ``ClauseSet`` holds integer clauses with a Symbol -> variable table; a
  false clause is ``{0}``.  ``add_clause`` adds one clause over literals;
  ``add_equivalence``, ``add_constraint`` and ``add_at_most_one`` encode an
  expression into clauses with an auxiliary variable per And/Or subterm
  (Tseitin), so the clause count stays linear in the size of the expression.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable


class Boolean:
    """Base of every expression."""

    __slots__ = ()


class BooleanAtom(Boolean):
    __slots__ = ()


class BooleanTrue(BooleanAtom):
    __slots__ = ()
    _instance: "BooleanTrue | None" = None

    def __new__(cls) -> "BooleanTrue":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return True

    def __eq__(self, other: Any) -> bool:
        return other is self or other is True

    def __hash__(self) -> int:
        return hash(True)

    def __repr__(self) -> str:
        return "True"


class BooleanFalse(BooleanAtom):
    __slots__ = ()
    _instance: "BooleanFalse | None" = None

    def __new__(cls) -> "BooleanFalse":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return False

    def __eq__(self, other: Any) -> bool:
        return other is self or other is False

    def __hash__(self) -> int:
        return hash(False)

    def __repr__(self) -> str:
        return "False"


true = BooleanTrue()
false = BooleanFalse()


class Symbol(BooleanAtom):
    """A named variable; equal to every Symbol of the same name."""

    __slots__ = ("name", "_hash")

    def __init__(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError(f"Symbol name must be a string, not {type(name)!r}")
        self.name = name
        self._hash = hash(("Symbol", name))

    def __eq__(self, other: Any) -> bool:
        return type(other) is Symbol and other.name == self.name

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return self.name

    __str__ = __repr__


class Dummy(Symbol):
    """A variable equal only to itself (an auxiliary variable)."""

    __slots__ = ()
    _count = itertools.count()

    def __init__(self, name: str = "Dummy") -> None:
        self.name = name
        self._hash = hash(("Dummy", next(Dummy._count)))

    def __eq__(self, other: Any) -> bool:
        return other is self

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return "_" + self.name

    __str__ = __repr__


def _as_bool(value: Any) -> Boolean:
    if isinstance(value, Boolean):
        return value
    if value is True:
        return true
    if value is False:
        return false
    raise ValueError(f"not a boolean expression: {value!r}")


class BooleanFunction(Boolean):
    """An operator node; ``args`` is a tuple of its operands."""

    __slots__ = ("args", "_key", "_hash")

    def __init__(self, *args: Any) -> None:
        self.args: tuple = tuple(args)
        self._key: Any = self.args
        self._hash = hash((type(self).__name__, self._key))

    def __eq__(self, other: Any) -> bool:
        return type(other) is type(self) and other._key == self._key

    def __hash__(self) -> int:
        return self._hash


class _Lattice(BooleanFunction):
    __slots__ = ()
    _identity: Boolean
    _zero: Boolean

    def __new__(cls, *args: Any) -> Any:
        flat: dict[Boolean, None] = {}
        stack = list(reversed(args))
        while stack:
            arg = _as_bool(stack.pop())
            if type(arg) is cls:
                stack.extend(reversed(arg.args))
                continue
            if arg is cls._zero:
                return cls._zero
            if arg is cls._identity:
                continue
            flat[arg] = None
        if not flat:
            return cls._identity
        if len(flat) == 1:
            return next(iter(flat))
        obj = object.__new__(cls)
        obj.args = tuple(flat)
        obj._key = frozenset(flat)
        obj._hash = hash((cls.__name__, obj._key))
        return obj

    def __init__(self, *args: Any) -> None:
        pass

    def __repr__(self) -> str:
        sep = " & " if type(self) is And else " | "
        return sep.join(
            f"({a!r})" if isinstance(a, _Lattice) else repr(a) for a in self.args
        )


class And(_Lattice):
    __slots__ = ()
    _identity = true
    _zero = false


class Or(_Lattice):
    __slots__ = ()
    _identity = false
    _zero = true


class Not(BooleanFunction):
    __slots__ = ()

    def __new__(cls, arg: Any) -> Any:
        arg = _as_bool(arg)
        if arg is true:
            return false
        if arg is false:
            return true
        if type(arg) is Not:
            return arg.args[0]
        obj = object.__new__(cls)
        obj.args = (arg,)
        obj._key = obj.args
        obj._hash = hash(("Not", obj._key))
        return obj

    def __init__(self, arg: Any) -> None:
        pass

    def __repr__(self) -> str:
        a = self.args[0]
        return f"~({a!r})" if isinstance(a, BooleanFunction) else f"~{a!r}"


class ClauseSet:
    """Integer clauses (sets of literals, ``0`` meaning False) and the
    Symbol -> variable number table."""

    __slots__ = ("data", "encoding")

    def __init__(
        self,
        data: list[set[int]] | None = None,
        encoding: dict[Any, int] | None = None,
    ) -> None:
        self.data: list[set[int]] = data if data is not None else []
        self.encoding: dict[Any, int] = encoding if encoding is not None else {}

    def copy(self) -> "ClauseSet":
        return ClauseSet([set(c) for c in self.data], dict(self.encoding))

    def literal(self, expr: Any) -> int | None:
        """The integer literal of a Symbol or its negation; 0 for False,
        None for True."""
        if expr is true:
            return None
        if expr is false:
            return 0
        if type(expr) is Not:
            n = self._var(expr.args[0])
            return -n
        return self._var(expr)

    def _var(self, symbol: Any) -> int:
        # a Symbol, or any other expression the encoders treat as opaque
        if not isinstance(symbol, Boolean) or isinstance(
            symbol, (_Lattice, Not, BooleanTrue, BooleanFalse)
        ):
            raise ValueError(f"not a literal: {symbol!r}")
        n = self.encoding.get(symbol)
        if n is None:
            n = self.encoding[symbol] = len(self.encoding) + 1
        return n

    def add_clause(self, literals: Iterable[Any]) -> None:
        """Add the disjunction of ``literals`` (Symbols, their negations or
        constants)."""
        clause: set[int] = set()
        for expr in literals:
            lit = self.literal(expr)
            if lit is None:
                return  # a True literal satisfies the clause
            if lit != 0:
                clause.add(lit)
        self.data.append(clause if clause else {0})


def add_equivalence(cnf: ClauseSet, symbol: Boolean, expr: Boolean) -> None:
    """Add ``symbol <-> expr`` to ``cnf`` without exponential growth.

    Every And/Or subterm of ``expr`` gets an auxiliary variable defined by a
    full equivalence (Tseitin encoding), which keeps the clause count linear
    in the size of ``expr`` (distributing And over Or would turn an Or of n
    Ands into 2^n clauses). Each auxiliary variable is fully determined by
    the original variables, so the satisfiability of ``cnf`` plus any
    constraint over the original variables is unchanged.

    ``symbol`` itself serves as the variable of the outermost And/Or, so a
    condition that is a single And/Or of literals costs no auxiliary variable
    at all.
    """
    literal = _tseitin(cnf, expr, {}, root=symbol)
    if literal is not symbol:
        cnf.add_clause([Not(symbol), literal])
        cnf.add_clause([symbol, Not(literal)])


def add_constraint(cnf: ClauseSet, expr: Boolean) -> None:
    """Add ``expr`` to ``cnf`` as a constraint without exponential growth.

    Like ``add_equivalence`` this gives every And/Or subterm an auxiliary
    variable, so an Or of n Ands costs a linear number of clauses.
    """
    cnf.add_clause([_tseitin(cnf, expr, {})])


def add_at_most_one(cnf: ClauseSet, literals: list[Any]) -> None:
    """Add "at most one of ``literals`` is true" to ``cnf``.

    Up to five literals get the pairwise clauses ``~a | ~b``. Longer
    lists get the binary encoding: ``ceil(log2 n)`` auxiliary bits, and
    each literal implies its own bit pattern, so two literals cannot both
    be true. That is ``n * ceil(log2 n)`` clauses instead of ``n(n-1)/2``.
    The bits are determined by the literals, so satisfiability with any
    constraint over the original variables is unchanged.
    """
    if len(literals) <= 5:
        for i, a in enumerate(literals):
            for b in literals[i + 1 :]:
                cnf.add_clause([Not(a), Not(b)])
        return
    width = (len(literals) - 1).bit_length()
    bits = [Dummy("at_most_one") for _ in range(width)]
    for code, literal in enumerate(literals):
        for k, bit in enumerate(bits):
            cnf.add_clause([Not(literal), bit if (code >> k) & 1 else Not(bit)])


def _tseitin(cnf: ClauseSet, expr: Any, cache: dict[Any, Any], root: Any = None) -> Any:
    """Return a literal (a Symbol, its negation or a constant) equivalent
    to ``expr``, adding defining clauses for any auxiliary variables to
    ``cnf``. When ``root`` is given and ``expr`` is an And/Or, ``root`` is
    used as its variable instead of a fresh one."""
    t = type(expr)
    if t is Not:
        return Not(_tseitin(cnf, expr.args[0], cache))
    if t is And or t is Or:
        if expr in cache:
            return cache[expr]
        literals = [_tseitin(cnf, arg, cache) for arg in expr.args]
        aux = Dummy("cond") if root is None else root
        if t is And:
            # aux -> each literal; all literals -> aux
            for literal in literals:
                cnf.add_clause([Not(aux), literal])
            cnf.add_clause([aux, *[Not(literal) for literal in literals]])
        else:
            # aux -> some literal; each literal -> aux
            cnf.add_clause([Not(aux), *literals])
            for literal in literals:
                cnf.add_clause([aux, Not(literal)])
        cache[expr] = aux
        return aux
    return expr
