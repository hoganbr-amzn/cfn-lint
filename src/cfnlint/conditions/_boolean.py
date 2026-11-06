"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0

Boolean expressions and a clause set for the condition engines.

The engines used sympy's logic module for two things: expression objects
(And/Or/Not/Implies over Symbols) that describe each condition, and
``EncodedCNF``, a list of integer clauses with a Symbol -> int table.  The
clauses themselves were always built by our own Tseitin and at-most-one
encodings (``cfnlint.conditions._utils``), so sympy contributed object
construction and its own distributive CNF conversion of small
propositions, and cost about 0.2 s of import per process.

This module keeps the part of sympy's behaviour the engines rely on:

* ``Symbol(name)`` is equal to every other ``Symbol`` with the same name,
  ``Dummy`` only to itself, and ``repr`` of either is what sympy gives
  (``name`` and ``_name``), so orders that sort by ``repr`` are unchanged.
* ``And``/``Or`` flatten nested operators of the same kind, drop the
  identity, short-circuit on the absorbing constant, drop duplicate
  arguments and compare equal regardless of argument order; with no
  argument they are the identity and with one they are that argument.
  ``Not`` removes double negation and negates constants.  ``Implies``
  simplifies on constants and on equal operands.  Complementary arguments
  (``x & ~x``) are kept, as sympy keeps them.
* Anything that is not an expression (``None``, a string) raises
  ``ValueError`` as sympy's ``SympifyError`` (a ``ValueError``) did.
* ``ClauseSet`` has the ``data``/``encoding``/``copy``/``add_prop``
  interface of ``EncodedCNF``; a false clause is ``{0}`` as in sympy.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable


class Boolean:
    """Base of every expression."""

    __slots__ = ()

    def __and__(self, other: Any) -> "Boolean":
        return And(self, other)

    def __rand__(self, other: Any) -> "Boolean":
        return And(other, self)

    def __or__(self, other: Any) -> "Boolean":
        return Or(self, other)

    def __ror__(self, other: Any) -> "Boolean":
        return Or(other, self)

    def __invert__(self) -> "Boolean":
        return Not(self)


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
        return type(other) is Symbol and other.name == self.name  # noqa: E721

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


class Implies(BooleanFunction):
    __slots__ = ()

    def __new__(cls, a: Any, b: Any) -> Any:
        a, b = _as_bool(a), _as_bool(b)
        if a is false or b is true:
            return true
        if a is true:
            return b
        if b is false:
            return Not(a)
        if a == b:
            return true
        obj = object.__new__(cls)
        obj.args = (a, b)
        obj._key = obj.args
        obj._hash = hash(("Implies", obj._key))
        return obj

    def __init__(self, a: Any, b: Any) -> None:
        pass

    def __repr__(self) -> str:
        return f"Implies({self.args[0]!r}, {self.args[1]!r})"


class ClauseSet:
    """Integer clauses (sets of literals, ``0`` meaning False) and the
    Symbol -> variable number table, as sympy's ``EncodedCNF``."""

    __slots__ = ("data", "encoding")

    def __init__(
        self,
        data: list[set[int]] | None = None,
        encoding: dict[Any, int] | None = None,
    ) -> None:
        self.data: list[set[int]] = data if data is not None else []
        self.encoding: dict[Any, int] = encoding if encoding is not None else {}

    @property
    def variables(self) -> list[int]:
        return list(range(1, len(self.encoding) + 1))

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
            symbol, (_Lattice, Not, Implies, BooleanTrue, BooleanFalse)
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

    def add_prop(self, expr: Any) -> None:
        """Add ``expr`` as a constraint.  And/Or/Not over literals become
        clauses directly; a nested Or argument gets a Tseitin variable."""
        self._add(_as_bool(expr), {})

    def _add(self, expr: Boolean, cache: dict) -> None:
        if expr is true:
            return
        t = type(expr)
        if t is And:
            for a in expr.args:
                self._add(a, cache)
            return
        if t is Or:
            self.add_clause(self._lit(a, cache) for a in expr.args)
            return
        if t is Not:
            inner = expr.args[0]
            ti = type(inner)
            if ti is Or:
                for a in inner.args:
                    self._add(Not(a), cache)
                return
            if ti is And:
                self.add_clause(self._lit(Not(a), cache) for a in inner.args)
                return
            if ti is Implies:
                self._add(inner.args[0], cache)
                self._add(Not(inner.args[1]), cache)
                return
        if t is Implies:
            self._add(Or(Not(expr.args[0]), expr.args[1]), cache)
            return
        self.add_clause([self._lit(expr, cache)])

    def _lit(self, expr: Boolean, cache: dict) -> Boolean:
        # imported here: _utils imports this module
        from cfnlint.conditions._utils import _tseitin

        return _tseitin(self, expr, cache)


__all__ = [
    "And",
    "Boolean",
    "BooleanAtom",
    "BooleanFalse",
    "BooleanFunction",
    "BooleanTrue",
    "ClauseSet",
    "Dummy",
    "Implies",
    "Not",
    "Or",
    "Symbol",
    "false",
    "true",
]
