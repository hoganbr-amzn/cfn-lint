"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""

import gc
import weakref

from cfnlint.config import ConfigMixIn
from cfnlint.decode import decode_str
from cfnlint.jsonschema._filter import FunctionFilter
from cfnlint.jsonschema.validators import CfnTemplateValidator
from cfnlint.rules.jsonschema.Base import BaseJsonSchema
from cfnlint.runner import Runner


class _Rule:
    def fail(self, validator, errors, instance, schema):
        yield from ()


def test_extend_without_a_cache_creates_a_class_every_time():
    rule = _Rule()
    first = CfnTemplateValidator({}).extend(validators={"fail": rule.fail})
    second = CfnTemplateValidator({}).extend(validators={"fail": rule.fail})
    assert first is not second


def test_extend_reuses_the_class_for_the_same_extension():
    cache = {}
    rule = _Rule()
    first = CfnTemplateValidator({}).extend(validators={"fail": rule.fail}, cache=cache)
    second = CfnTemplateValidator({}).extend(
        validators={"fail": rule.fail}, cache=cache
    )
    assert first is second
    assert len(cache) == 1

    other = CfnTemplateValidator({}).extend(
        validators={"fail": _Rule().fail}, cache=cache
    )
    assert other is not first

    without_keyword = CfnTemplateValidator({}).extend(
        validators={"fail": rule.fail},
        function_filter=FunctionFilter(add_cfn_lint_keyword=False),
        cache=cache,
    )
    assert without_keyword is not first
    assert without_keyword({}).function_filter.add_cfn_lint_keyword is False
    assert first({}).function_filter.add_cfn_lint_keyword is True
    assert len(cache) == 3


def test_extend_with_unhashable_validators_is_not_cached():
    class _Unhashable:
        __hash__ = None  # type: ignore

        def __call__(self, validator, errors, instance, schema):
            yield from ()

    cache = {}
    first = CfnTemplateValidator({}).extend(
        validators={"fail": _Unhashable()}, cache=cache
    )
    second = CfnTemplateValidator({}).extend(
        validators={"fail": _Unhashable()}, cache=cache
    )
    assert first is not second
    assert cache == {}


def test_rule_keeps_its_extended_classes_only_while_it_lives():
    class _SchemaRule(BaseJsonSchema):
        id = "E9999"

        def fail(self, validator, errors, instance, schema):
            yield from ()

    rule = _SchemaRule()
    rule.validators["fail"] = rule.fail
    validator = CfnTemplateValidator({})
    first = rule.extend_validator(validator, {}, validator.context)
    second = rule.extend_validator(validator, {}, validator.context)
    assert type(first) is type(second)
    assert len(rule._extended_validators) == 1

    ref = weakref.ref(rule)
    del rule, first, second
    gc.collect()
    assert ref() is None


def test_many_lint_runs_in_one_process_release_their_rules():
    template, _ = decode_str(
        "Resources:\n"
        "  Bucket:\n"
        "    Type: AWS::S3::Bucket\n"
        "    Properties:\n"
        "      BucketName: bucket\n"
        "      Tags: [{Key: a, Value: b}]\n"
    )
    rule_refs = []
    for _ in range(3):
        runner = Runner(ConfigMixIn(regions=["us-east-1"]))
        list(runner.validate_template(template))
        rules = list(runner.rules.values())
        # the rules extended validators while linting
        assert any(getattr(rule, "_extended_validators", None) for rule in rules)
        rule_refs.append([weakref.ref(rule) for rule in rules])
        del runner, rules
        gc.collect()

    # nothing from a finished lint is kept alive by a later one
    alive = [sum(ref() is not None for ref in refs) for refs in rule_refs]
    assert alive == [0, 0, 0], alive
