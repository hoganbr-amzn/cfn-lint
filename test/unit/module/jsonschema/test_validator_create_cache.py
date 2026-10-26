"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

from dataclasses import dataclass

import pytest

from cfnlint.jsonschema import _keywords
from cfnlint.jsonschema import validators as validators_module
from cfnlint.jsonschema._filter import FunctionFilter
from cfnlint.jsonschema.validators import CfnTemplateValidator, create


@pytest.fixture(autouse=True)
def clear_cache():
    validators_module._validator_classes.clear()
    yield
    validators_module._validator_classes.clear()


def test_same_arguments_return_the_same_class():
    first = create(
        validators={"type": _keywords.type}, function_filter=FunctionFilter()
    )
    second = create(
        validators={"type": _keywords.type}, function_filter=FunctionFilter()
    )

    assert first is second


def test_extend_reuses_the_class():
    validator = CfnTemplateValidator({})

    first = validator.extend(validators={"type": _keywords.type})
    second = validator.extend(validators={"type": _keywords.type})

    assert first is second
    # instances built from the shared class stay independent
    assert first({"type": "string"}).schema == {"type": "string"}
    assert second({"type": "number"}).schema == {"type": "number"}


def test_different_arguments_return_different_classes():
    base = create(function_filter=FunctionFilter())

    assert (
        create(function_filter=FunctionFilter(add_cfn_lint_keyword=False)) is not base
    )
    assert (
        create(function_filter=FunctionFilter(), validators={"type": _keywords.type})
        is not base
    )
    assert (
        create(
            function_filter=FunctionFilter(),
            fn_resolvers={"Ref": lambda v, i: iter(())},
        )
        is not base
    )


def test_function_filter_subclass_gets_its_own_class():
    @dataclass
    class CustomFilter(FunctionFilter):
        pass

    assert create(function_filter=CustomFilter()) is not create(
        function_filter=FunctionFilter()
    )


def test_unhashable_arguments_build_without_caching():
    class Unhashable:
        __hash__ = None  # type: ignore[assignment]

        def __call__(self, validator, value, instance, schema):
            yield from ()

    cls = create(validators={"custom": Unhashable()})

    assert list(cls({"custom": True}).iter_errors("a")) == []
    assert not validators_module._validator_classes


def test_cache_evicts_the_least_recently_used_class(monkeypatch):
    monkeypatch.setattr(validators_module, "_MAX_VALIDATOR_CLASSES", 2)

    first = create(function_filter=FunctionFilter(add_cfn_lint_keyword=True))
    second = create(function_filter=FunctionFilter(add_cfn_lint_keyword=False))
    # hitting first makes second the least recently used
    assert create(function_filter=FunctionFilter(add_cfn_lint_keyword=True)) is first
    third = create(function_filter=FunctionFilter(validate_dynamic_references=False))

    assert len(validators_module._validator_classes) == 2
    assert create(function_filter=FunctionFilter(add_cfn_lint_keyword=True)) is first
    assert (
        create(function_filter=FunctionFilter(validate_dynamic_references=False))
        is third
    )
    assert (
        create(function_filter=FunctionFilter(add_cfn_lint_keyword=False)) is not second
    )
