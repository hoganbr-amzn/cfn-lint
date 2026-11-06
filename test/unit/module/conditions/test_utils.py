"""
Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

import hashlib
import itertools
import json
import unittest

from cfnlint.conditions._boolean import (
    And,
    BooleanFalse,
    BooleanFunction,
    BooleanTrue,
    ClauseSet,
    Not,
    Or,
    Symbol,
    add_equivalence,
)
from cfnlint.conditions._utils import (
    ObjectEncoder,
    _cached_hash,
    _make_hashable,
    cfn_string,
    get_hash,
)


class AnObject:
    """Test object with _value attribute"""

    def __init__(self, value):
        self._value = value


class BadStringObject:
    """Test object that raises exception when stringified"""

    def __str__(self):
        raise ValueError("Cannot stringify")


class TestUtils(unittest.TestCase):
    """Test Utils"""

    def test_object_encoder(self):
        """Test ObjectEncoder class"""
        # Test with a regular object
        test_obj = AnObject("test_value")
        encoded = json.dumps(test_obj, cls=ObjectEncoder)
        self.assertEqual(encoded, '"test_value"')

        # Test with a nested object
        nested_obj = AnObject(AnObject("nested_value"))
        encoded = json.dumps(nested_obj, cls=ObjectEncoder)
        self.assertEqual(encoded, '"nested_value"')

        # Test with a regular value
        regular_value = {"key": "value"}
        encoded = json.dumps(regular_value, cls=ObjectEncoder)
        self.assertEqual(encoded, '{"key": "value"}')

        # Test the default method directly
        encoder = ObjectEncoder()
        obj = object()
        # The default method should return the object itself
        self.assertEqual(encoder.default(obj), obj)

    def test_make_hashable_simple_types(self):
        """Test _make_hashable with simple types"""
        # Test with simple hashable types
        self.assertEqual(_make_hashable("string"), "string")
        self.assertEqual(_make_hashable(123), 123)
        self.assertEqual(_make_hashable(123.45), 123.45)
        self.assertEqual(_make_hashable(True), (bool, True))
        self.assertEqual(_make_hashable(None), None)

    def test_booleans_hash_apart_from_integers(self):
        """True == 1 and False == 0 in Python, so an lru_cache keyed by the
        hashable form returned the first caller's digest for both. CloudFormation
        compares true and 1 as the strings "true" and "1", which differ."""
        for a, b in ((True, 1), (False, 0)):
            self.assertNotEqual(get_hash(a), get_hash(b))
            self.assertNotEqual(get_hash(b), get_hash(a))
            self.assertNotEqual(
                get_hash([{"Ref": "P"}, a]), get_hash([{"Ref": "P"}, b])
            )
            self.assertNotEqual(
                get_hash([{"Ref": "P"}, b]), get_hash([{"Ref": "P"}, a])
            )
        # the same value still hashes the same, whatever was asked first
        self.assertEqual(get_hash([{"Ref": "P"}, True]), get_hash([{"Ref": "P"}, True]))
        self.assertEqual(get_hash(1), get_hash(1))

    def test_cfn_string(self):
        """Literals become the strings CloudFormation compares."""
        self.assertEqual(cfn_string(True), "true")
        self.assertEqual(cfn_string(False), "false")
        self.assertEqual(cfn_string("True"), "True")
        self.assertEqual(cfn_string("true"), "true")
        self.assertEqual(cfn_string(1), "1")
        self.assertEqual(cfn_string(0), "0")
        self.assertEqual(cfn_string(1.0), "1.0")
        self.assertEqual(cfn_string("1"), "1")
        self.assertEqual(cfn_string({"Ref": "P"}), {"Ref": "P"})
        self.assertEqual(cfn_string(None), None)

    def test_make_hashable_dict(self):
        """Test _make_hashable with dictionaries"""
        # Test with dictionary
        test_dict = {"b": 2, "a": 1}
        hashable_dict = _make_hashable(test_dict)
        # Should be a tuple of tuples with sorted keys
        self.assertEqual(hashable_dict, (("a", 1), ("b", 2)))

        # Test with nested dictionary
        nested_dict = {"outer": {"inner": "value"}}
        hashable_nested = _make_hashable(nested_dict)
        self.assertEqual(hashable_nested, (("outer", (("inner", "value"),)),))

    def test_make_hashable_list(self):
        """Test _make_hashable with lists"""
        # Test with list
        test_list = [1, 2, 3]
        hashable_list = _make_hashable(test_list)
        self.assertEqual(hashable_list, (1, 2, 3))

        # Test with nested list
        nested_list = [1, [2, 3], 4]
        hashable_nested = _make_hashable(nested_list)
        self.assertEqual(hashable_nested, (1, (2, 3), 4))

    def test_make_hashable_complex(self):
        """Test _make_hashable with complex nested structures"""
        # Test with complex nested structure
        complex_obj = {
            "list": [1, {"key": "value"}],
            "dict": {"nested": [1, 2, 3]},
        }
        hashable_complex = _make_hashable(complex_obj)
        expected = (
            ("dict", (("nested", (1, 2, 3)),)),
            ("list", (1, (("key", "value"),))),
        )
        self.assertEqual(hashable_complex, expected)

    def test_make_hashable_with_object(self):
        """Test _make_hashable with custom objects"""
        # Test with object having _value
        test_obj = AnObject("test_value")
        hashable_obj = _make_hashable(test_obj)
        self.assertEqual(hashable_obj, "test_value")

        # Test with nested object
        nested_obj = AnObject({"key": "value"})
        hashable_nested = _make_hashable(nested_obj)
        self.assertEqual(hashable_nested, (("key", "value"),))

    def test_cached_hash_simple_types(self):
        """Test _cached_hash with simple types"""
        # Test with simple types
        string_hash = _cached_hash("test")
        expected_hash = hashlib.sha1("test".encode("utf-8")).hexdigest()
        self.assertEqual(string_hash, expected_hash)

        int_hash = _cached_hash(123)
        expected_hash = hashlib.sha1("123".encode("utf-8")).hexdigest()
        self.assertEqual(int_hash, expected_hash)

    def test_cached_hash_complex_types(self):
        """Test _cached_hash with complex types"""
        # Test with tuple (hashable)
        tuple_value = (1, 2, 3)
        tuple_hash = _cached_hash(tuple_value)
        expected_hash = hashlib.sha1(str(tuple_value).encode("utf-8")).hexdigest()
        self.assertEqual(tuple_hash, expected_hash)

    def test_cached_hash_exception_handling(self):
        """Test _cached_hash exception handling"""
        # Create a simple object that will trigger the exception path
        bad_obj = BadStringObject()

        # Mock the json.dumps to avoid circular reference issues
        original_dumps = json.dumps

        try:
            # Replace json.dumps with a mock that returns a fixed string
            def mock_dumps(*args, **kwargs):
                return '{"mocked": "value"}'

            json.dumps = mock_dumps

            # This should now use our mocked json.dumps in the exception handler
            hash_result = _cached_hash(bad_obj)

            # Verify we got a hash (the exact value doesn't matter)
            self.assertTrue(isinstance(hash_result, str))
            self.assertEqual(len(hash_result), 40)  # SHA-1 hash is 40 chars
        finally:
            # Restore the original json.dumps
            json.dumps = original_dumps

    def test_get_hash_simple_types(self):
        """Test get_hash with simple types"""
        # Test with simple types
        string_hash = get_hash("test")
        expected_hash = hashlib.sha1("test".encode("utf-8")).hexdigest()
        self.assertEqual(string_hash, expected_hash)

        int_hash = get_hash(123)
        expected_hash = hashlib.sha1("123".encode("utf-8")).hexdigest()
        self.assertEqual(int_hash, expected_hash)

    def test_get_hash_dict(self):
        """Test get_hash with dictionaries"""
        # Test with dictionary
        test_dict = {"b": 2, "a": 1}
        dict_hash = get_hash(test_dict)

        # Create the same dictionary in different order
        test_dict2 = {"a": 1, "b": 2}
        dict_hash2 = get_hash(test_dict2)

        # Hashes should be the same regardless of key order
        self.assertEqual(dict_hash, dict_hash2)

    def test_get_hash_list(self):
        """Test get_hash with lists"""
        # Test with list
        test_list = [1, 2, 3]
        list_hash = get_hash(test_list)

        # The same list should produce the same hash
        list_hash2 = get_hash([1, 2, 3])
        self.assertEqual(list_hash, list_hash2)

    def test_get_hash_complex(self):
        """Test get_hash with complex nested structures"""
        # Test with complex nested structure
        complex_obj = {
            "list": [1, {"key": "value"}],
            "dict": {"nested": [1, 2, 3]},
        }
        complex_hash = get_hash(complex_obj)

        # Different order but same content
        complex_obj2 = {
            "dict": {"nested": [1, 2, 3]},
            "list": [1, {"key": "value"}],
        }
        complex_hash2 = get_hash(complex_obj2)

        # Hashes should be the same
        self.assertEqual(complex_hash, complex_hash2)

    def test_get_hash_with_object(self):
        """Test get_hash with custom objects"""
        # Test with object having _value
        test_obj = AnObject("test_value")
        obj_hash = get_hash(test_obj)

        # Should be the same as hashing the _value directly
        direct_hash = get_hash("test_value")
        self.assertEqual(obj_hash, direct_hash)

    def test_get_hash_exception_handling(self):
        """Test get_hash exception handling"""
        # Create a simple object that will trigger the exception path
        bad_obj = BadStringObject()

        # Mock the json.dumps to avoid circular reference issues
        original_dumps = json.dumps

        try:
            # Replace json.dumps with a mock that returns a fixed string
            def mock_dumps(*args, **kwargs):
                return '{"mocked": "value"}'

            json.dumps = mock_dumps

            # This should now use our mocked json.dumps in the exception handler
            hash_result = get_hash(bad_obj)

            # Verify we got a hash (the exact value doesn't matter)
            self.assertTrue(isinstance(hash_result, str))
            self.assertEqual(len(hash_result), 40)  # SHA-1 hash is 40 chars
        finally:
            # Restore the original json.dumps
            json.dumps = original_dumps

    def test_get_hash_make_hashable_exception(self):
        """Test get_hash when _make_hashable raises an exception"""
        # Create a mock for _make_hashable that raises an exception
        original_make_hashable = _make_hashable

        try:
            # Replace _make_hashable with a function that raises an exception
            def mock_make_hashable(obj):
                raise Exception("Cannot make hashable")

            # Monkey patch the function
            import cfnlint.conditions._utils

            cfnlint.conditions._utils._make_hashable = mock_make_hashable

            # Mock json.dumps to avoid circular reference issues
            original_dumps = json.dumps
            json.dumps = lambda *args, **kwargs: '{"mocked": "value"}'

            # This should trigger the fallback path in get_hash
            obj = {"test": "value"}
            hash_result = get_hash(obj)

            # Verify we got a hash
            self.assertTrue(isinstance(hash_result, str))
            self.assertEqual(len(hash_result), 40)  # SHA-1 hash is 40 chars
        finally:
            # Restore the original functions
            cfnlint.conditions._utils._make_hashable = original_make_hashable
            json.dumps = original_dumps

    def test_hash_consistency(self):
        """Test hash consistency across multiple calls"""
        # Test that the same object produces the same hash on multiple calls
        test_obj = {"complex": [1, 2, {"nested": "value"}]}

        hash1 = get_hash(test_obj)
        hash2 = get_hash(test_obj)
        hash3 = get_hash(test_obj)

        self.assertEqual(hash1, hash2)
        self.assertEqual(hash2, hash3)

    def test_lru_cache_effectiveness(self):
        """Test that the LRU cache is effective"""
        # Call get_hash multiple times with the same object
        test_obj = {"test": "value"}

        # First call should compute the hash
        hash1 = get_hash(test_obj)

        # Subsequent calls should use the cached value
        hash2 = get_hash(test_obj)

        self.assertEqual(hash1, hash2)

        # We can't directly test the cache hit, but we can
        # verify the result is consistent


def _value(expr, values):
    """Truth value of ``expr`` under ``values`` (Symbol -> bool)"""
    if isinstance(expr, BooleanTrue):
        return True
    if isinstance(expr, BooleanFalse):
        return False
    if isinstance(expr, Not):
        return not _value(expr.args[0], values)
    if isinstance(expr, And):
        return all(_value(a, values) for a in expr.args)
    if isinstance(expr, Or):
        return any(_value(a, values) for a in expr.args)
    return values[expr]


def _satisfiable(cnf: ClauseSet, pins) -> bool:
    """Brute force: is ``cnf`` satisfiable with ``pins`` (Symbol, bool)"""
    fixed = {cnf.encoding[s]: v for s, v in pins if s in cnf.encoding}
    free = [n for n in range(1, len(cnf.encoding) + 1) if n not in fixed]
    for bits in itertools.product([True, False], repeat=len(free)):
        model = dict(fixed)
        model.update(zip(free, bits))
        if all(
            any(lit != 0 and model[abs(lit)] == (lit > 0) for lit in clause)
            for clause in cnf.data
        ):
            return True
    return False


class TestAddEquivalence(unittest.TestCase):
    """Test the Tseitin encoding used for condition equivalences"""

    def test_matches_direct_encoding(self):
        """For every assignment of the original variables the encoded
        equivalence is satisfiable exactly when the symbol takes the value
        of the expression"""
        a, b, c, d = Symbol("a"), Symbol("b"), Symbol("c"), Symbol("d")
        sym = Symbol("cond")
        expressions = [
            a,
            Not(a),
            And(a, b),
            Or(a, b),
            Or(And(a, b), And(c, d), Not(Or(a, d))),
            Not(And(Or(a, b), Not(c))),
            Or(Not(And(a, b)), Or(c, d)),
            # a <-> (b | c)
            And(Or(Not(a), Or(b, c)), Or(Not(Or(b, c)), a)),
            # a xor b xor c
            Or(
                And(a, Not(b), Not(c)),
                And(Not(a), b, Not(c)),
                And(Not(a), Not(b), c),
                And(a, b, c),
            ),
            And(Or(a, b), Or(a, b), c),
            BooleanTrue(),
            BooleanFalse(),
        ]
        for expr in expressions:
            encoded = ClauseSet()
            add_equivalence(encoded, sym, expr)
            for values in itertools.product([True, False], repeat=5):
                assignment = dict(zip([a, b, c, d, sym], values))
                self.assertEqual(
                    _value(expr, assignment) == assignment[sym],
                    _satisfiable(encoded, assignment.items()),
                    f"{expr} with {assignment}",
                )

    def test_or_of_ands_is_linear(self):
        """An Or of n Ands is 2^n clauses when distributed, but linear
        when each subterm gets its own variable"""
        s = Symbol("s")
        expr = Or(*[And(Symbol(f"r{i}"), s) for i in range(40)])
        cnf = ClauseSet()
        add_equivalence(cnf, Symbol("cond"), expr)
        self.assertLess(len(cnf.data), 200)

    def test_single_and_or_needs_no_auxiliary_variable(self):
        """The condition symbol is the variable of the outermost And/Or, so
        the common one-level condition encodes exactly like the direct
        equivalence"""
        a, b, cond = Symbol("a"), Symbol("b"), Symbol("cond")
        direct = {
            "and": [
                {(cond, False), (a, True)},
                {(cond, False), (b, True)},
                {(cond, True), (a, False), (b, False)},
            ],
            "or": [
                {(cond, False), (a, True), (b, True)},
                {(cond, True), (a, False)},
                {(cond, True), (b, False)},
            ],
        }
        for name, expr in [("and", And(a, b)), ("or", Or(a, b))]:
            encoded = ClauseSet()
            add_equivalence(encoded, cond, expr)
            self.assertCountEqual(encoded.encoding, [a, b, cond])
            self.assertCountEqual(
                _decode(encoded), [frozenset(c) for c in direct[name]]
            )

    def test_repeated_subterm_shares_one_variable(self):
        """A subterm that appears under several parents is encoded once and
        its variable reused"""
        a, b, c, d = Symbol("a"), Symbol("b"), Symbol("c"), Symbol("d")
        sym = Symbol("cond")
        shared = Or(a, b)
        expr = Or(And(shared, c), And(shared, d))
        encoded = ClauseSet()
        add_equivalence(encoded, sym, expr)
        # a, b, c, d, cond plus one variable each for the two Ands and one
        # for the shared Or -- not a second copy of the shared Or
        self.assertEqual(len(encoded.encoding), 8)
        for values in itertools.product([True, False], repeat=5):
            assignment = dict(zip([a, b, c, d, sym], values))
            self.assertEqual(
                _value(expr, assignment) == assignment[sym],
                _satisfiable(encoded, assignment.items()),
            )

    def test_unknown_boolean_function_is_a_literal(self):
        """A BooleanFunction the encoder does not know is treated as an
        opaque literal instead of recursing forever"""

        class Opaque(BooleanFunction):
            pass

        a, b, sym = Symbol("a"), Symbol("b"), Symbol("cond")
        opaque = Opaque(a)
        expr = And(opaque, b)
        encoded = ClauseSet()
        add_equivalence(encoded, sym, expr)
        for values in itertools.product([True, False], repeat=3):
            pins = list(zip([opaque, b, sym], values))
            self.assertEqual(
                _satisfiable(encoded, pins), values[2] == (values[0] and values[1])
            )


def _decode(cnf: ClauseSet) -> list[frozenset]:
    """Clauses of ``cnf`` as (symbol, is_positive) pairs"""
    symbols = {n: s for s, n in cnf.encoding.items()}
    return [frozenset((symbols[abs(i)], i > 0) for i in clause) for clause in cnf.data]


if __name__ == "__main__":
    unittest.main()
