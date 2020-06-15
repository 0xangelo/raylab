# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
import torch.nn as nn

from raylab.losses import ModelEnsembleMLE
from raylab.modules.mixins.stochastic_model_mixin import StochasticModelMixin
from raylab.utils.debug import fake_batch


@pytest.fixture(params=(1, 2, 4), ids=(f"Models({n})" for n in (1, 2, 4)))
def models(request, obs_space, action_space):
    config = {
        "encoder": {"units": (32,)},
        "residual": True,
        "input_dependent_scale": True,
    }

    def model():
        return StochasticModelMixin.build_single_model(obs_space, action_space, config)

    return nn.ModuleList([model() for _ in range(request.param)])


@pytest.fixture
def loss_fn(models):
    return ModelEnsembleMLE(models)


def test_init(loss_fn):
    assert hasattr(loss_fn, "batch_keys")


@pytest.fixture(scope="module")
def batch(obs_space, action_space):
    samples = fake_batch(obs_space, action_space, batch_size=256)
    return {k: torch.from_numpy(v) for k, v in samples.items()}


def test_call(loss_fn, batch):
    loss, info = loss_fn(batch)

    assert torch.is_tensor(loss)
    assert isinstance(info, dict)
    assert all(isinstance(k, str) for k in info.keys())
    assert all(isinstance(v, float) for v in info.values())


def test_compile(loss_fn, batch):
    loss_fn.compile()
    loss, _ = loss_fn(batch)

    assert torch.is_tensor(loss)
