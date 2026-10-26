"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

from dataclasses import fields

import pytest

from cfnlint.jsonschema.validators import CfnTemplateValidator


@pytest.fixture
def validator():
    return CfnTemplateValidator({"type": "string"})


def test_evolve_matches_init(validator):
    evolved = validator.evolve(schema={"type": "number"})

    assert type(evolved) is type(validator)
    assert evolved.schema == {"type": "number"}
    assert evolved.context is validator.context
    assert evolved.resolver is validator.resolver
    assert evolved.cfn is validator.cfn
    assert evolved.function_filter is validator.function_filter
    assert evolved.validators is validator.validators
    assert evolved.fn_resolvers is validator.fn_resolvers
    assert evolved.format_checker is validator.format_checker
    assert evolved._type_checker is validator._type_checker
    assert validator.schema == {"type": "string"}


def test_evolve_resets_fields_init_does_not_take(validator):
    # __init__ gives these fields their defaults regardless of the source
    # instance, so a value set on the instance must not carry over
    names = [f.name for f in fields(type(validator)) if not f.init]
    assert names
    for name in names:
        setattr(validator, name, object())

    evolved = validator.evolve(schema={"type": "number"})

    fresh = CfnTemplateValidator({})
    for name in names:
        assert getattr(evolved, name) is getattr(fresh, name)


def test_evolve_rejects_unknown_keyword(validator):
    with pytest.raises(TypeError):
        validator.evolve(not_a_field=True)


def test_evolve_rebuilds_a_missing_resolver(validator):
    evolved = validator.evolve(schema={"type": "number"}, resolver=None)

    assert evolved.resolver is not None
    assert evolved.resolver is not validator.resolver


def test_evolve_rebuilds_a_missing_context(validator):
    evolved = validator.evolve(context=None)

    assert evolved.context is not None
    assert evolved.context == validator.cfn.context
