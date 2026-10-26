"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

from collections import deque
from dataclasses import fields
from functools import cached_property

import pytest

from cfnlint.context.context import Context, Path
from cfnlint.template import Template


@pytest.fixture
def context():
    template = Template(
        None,
        {
            "Parameters": {"MyParameter": {"Type": "String"}},
            "Resources": {
                "MyBucket": {"Type": "AWS::S3::Bucket"},
                "MyModule": {"Type": "My::Module::MODULE"},
            },
        },
        ["us-east-1"],
    )
    return template.context


def test_context_init_sets_every_field():
    # Context.evolve copies the instance dict instead of calling __init__,
    # which only holds while every field is set by __init__ and nothing
    # runs after it
    assert all(f.init for f in fields(Context))
    assert not hasattr(Context, "__post_init__")


def test_context_module_names_is_the_only_cached_property():
    # the copied dict carries cached values along, and evolve drops
    # module_names when resources change; a new cached_property needs
    # the same treatment for the fields it reads
    cached = [
        name
        for name, attr in vars(Context).items()
        if isinstance(attr, cached_property)
    ]
    assert cached == ["module_names"]


def test_context_evolve_matches_init(context):
    before = {f.name: getattr(context, f.name) for f in fields(Context)}

    evolved = context.evolve(functions=["Ref"], strict_types=False)
    built = Context(**{**before, "functions": ["Ref"], "strict_types": False})

    assert evolved == built
    assert evolved is not context
    assert {f.name: getattr(context, f.name) for f in fields(Context)} == before


def test_context_evolve_merges_ref_values(context):
    first = context.evolve(ref_values={"A": "1"})
    second = first.evolve(ref_values={"B": "2"})

    assert second.ref_values == {"A": "1", "B": "2"}
    assert first.ref_values == {"A": "1"}
    assert context.ref_values == {}


def test_context_evolve_rejects_unknown_keyword(context):
    with pytest.raises(TypeError):
        context.evolve(not_a_field=True)


def test_path_descend_appends_without_sharing():
    path = Path(path=deque(["Resources"]), cfn_path=deque(["Resources"]))

    descended = path.descend(path="MyBucket", cfn_path="AWS::S3::Bucket")

    assert descended.path == deque(["Resources", "MyBucket"])
    assert descended.cfn_path == deque(["Resources", "AWS::S3::Bucket"])
    assert descended.value_path == deque([])
    assert path.path == deque(["Resources"])
    assert path.cfn_path == deque(["Resources"])


def test_path_evolve_replaces_without_appending():
    path = Path(path=deque(["Resources"]))

    evolved = path.evolve(path=deque(["Outputs"]))

    assert evolved.path == deque(["Outputs"])
    assert path.path == deque(["Resources"])
