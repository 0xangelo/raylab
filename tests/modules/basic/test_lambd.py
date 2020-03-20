# pylint: disable=missing-docstring,redefined-outer-name,protected-access,invalid-name
import pytest
import torch
import torch.distributions as ptd

from raylab.modules.basic import Lambda


def categorical_sample(logits):
    return ptd.Categorical(logits=logits).sample()


traced_categorical_sample = torch.jit.trace(
    categorical_sample, torch.ones(4), check_trace=False
)


def exp(inputs):
    return inputs.exp()


scripted_exp = torch.jit.script(exp)


def randn_like(inputs):
    return torch.randn_like(inputs)


scripted_randn_like = torch.jit.script(randn_like)


FUNCS = (
    categorical_sample,
    exp,
    randn_like,
)

SCRIPT_ARGS = (
    pytest.param(categorical_sample, marks=pytest.mark.xfail,),
    pytest.param(exp, marks=pytest.mark.xfail,),
    traced_categorical_sample,
    scripted_exp,
    scripted_randn_like,
)


@pytest.fixture(params=FUNCS)
def func(request):
    return request.param


def test_eager(func):
    module = Lambda(func)

    inputs = torch.randn(2, 4)
    module(inputs)


@pytest.fixture(
    params=SCRIPT_ARGS,
    ids=("FailDist", "FailExp", "TracedDist", "ScriptedExp", "ScriptedRandnLike"),
)
def script_func(request):
    return request.param


def test_script(script_func):
    module = torch.jit.script(Lambda(script_func))

    inputs = torch.randn(4)
    module(inputs)
