# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch

from raylab.modules.basic import DiagMultivariateNormalParams


@pytest.fixture(params=(True, False))
def torch_script(request):
    return request.param


@pytest.fixture(params=(True, False))
def input_dependent_scale(request):
    return request.param


@pytest.fixture
def module_and_logits_fn(torch_script):
    cls = DiagMultivariateNormalParams
    maker = cls.as_script_module if torch_script else cls

    def func(input_dependent_scale):
        module = maker(
            in_features=10, event_dim=4, input_dependent_scale=input_dependent_scale
        )
        return module, torch.randn(6, 10, requires_grad=True)

    return func


def test_output_shape(module_and_logits_fn, input_dependent_scale):
    module, logits = module_and_logits_fn(input_dependent_scale)

    params = module(logits)
    assert "loc" in params
    assert "scale_diag" in params
    assert params["loc"].dtype == torch.float32
    assert params["scale_diag"].dtype == torch.float32
    assert params["loc"].shape == (6, 4)
    assert params["scale_diag"].shape == (6, 4)


def test_gradient_propagation(module_and_logits_fn):
    module, logits = module_and_logits_fn(input_dependent_scale)

    params = module(logits)
    params["loc"].mean().backward()
    assert logits.grad is not None

    logits.grad = None
    params = module(logits)
    params["scale_diag"].mean().backward()
    assert not input_dependent_scale or logits.grad is not None
