"""
Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

import functools
import hashlib
import json
from typing import Any, Hashable


class ObjectEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if hasattr(o, "_value"):
            # pylint: disable=protected-access
            return o._value
        return o


def cfn_string(value: Any) -> Any:
    """Return a scalar literal as the string CloudFormation compares.

    CloudFormation compares Fn::Equals operands, parameter values and
    AllowedValues as strings, and writes a template boolean as ``true`` or
    ``false`` (Python's ``str`` gives ``True``/``False``). Numbers keep
    Python's ``str`` (``1`` is "1", ``1.0`` is "1.0"). Anything that is not
    a bool, int or float (strings, dicts, lists, None) is returned unchanged.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return value


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
