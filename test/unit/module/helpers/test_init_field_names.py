"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

from dataclasses import dataclass, field, fields

from cfnlint.context import Context
from cfnlint.helpers import init_field_names


def test_init_field_names_matches_the_dataclass_fields():
    names = init_field_names(Context)

    assert names == tuple(f.name for f in fields(Context) if f.init)
    assert "path" in names
    assert "transforms" in names


def test_init_field_names_is_cached_on_the_class():
    @dataclass
    class Base:
        a: int = 0
        b: int = field(init=False, default=0)

    @dataclass
    class Child(Base):
        c: int = 0

    assert init_field_names(Base) == ("a",)
    assert Base.__dict__["_init_field_names"] == ("a",)
    assert init_field_names(Base) is init_field_names(Base)

    # a subclass gets its own entry instead of inheriting the parent's
    assert init_field_names(Child) == ("a", "c")
    assert init_field_names(Base) == ("a",)
