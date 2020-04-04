# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
import torch.nn as nn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import merge_dicts

from raylab.modules.deterministic_actor_mixin import DeterministicActorMixin


BASE_CONFIG = {
    "torch_script": False,
    "double_q": False,
    "exploration": None,
    "exploration_gaussian_sigma": 0.3,
    "smooth_target_policy": False,
    "target_gaussian_sigma": 0.3,
    "actor": {
        "units": (32, 32),
        "activation": "ReLU",
        "initializer_options": {"name": "xavier_uniform"},
        "beta": 1.2,
    },
    "critic": {
        "units": (32, 32),
        "activation": "ReLU",
        "initializer_options": {"name": "xavier_uniform"},
        "delay_action": True,
    },
}


class DummyModule(DeterministicActorMixin, nn.ModuleDict):
    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        super().__init__()
        self.update(
            self._make_actor(obs_space, action_space, merge_dicts(BASE_CONFIG, config))
        )


@pytest.fixture(scope="module", params=(DummyModule,))
def module_cls(request):
    return request.param


@pytest.fixture(scope="module", params=(None, "gaussian", "parameter_noise"))
def exploration(request):
    return request.param


@pytest.fixture(scope="module", params=(0.3, 0.0))
def exploration_gaussian_sigma(request):
    return request.param


@pytest.fixture(scope="module", params=(0.8, 1.2))
def beta(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=(True, False),
    ids=("Smooth Target Policy", "Hard Target Policy"),
)
def smooth_target_policy(request):
    return request.param


@pytest.fixture(scope="module")
def full_config(exploration, exploration_gaussian_sigma, beta, smooth_target_policy):
    return {
        "exploration": exploration,
        "exploration_gaussian_sigma": exploration_gaussian_sigma,
        "smooth_target_policy": smooth_target_policy,
        "actor": {"beta": beta},
    }


@pytest.fixture(scope="module")
def module_batch_config(module_and_batch_fn, module_cls, full_config):
    module, batch = module_and_batch_fn(module_cls, full_config)
    return module, batch, full_config


def test_module_creation(module_batch_config):
    module, _, _ = module_batch_config

    assert "actor" in module
    actor = module.actor
    assert "policy" in module.actor
    assert "behavior" in module.actor
    assert "target_policy" in module.actor
    assert all(
        torch.allclose(p, p_)
        for p, p_ in zip(actor.policy.parameters(), actor.target_policy.parameters())
    )


def test_policy(module_batch_config):
    module, batch, config = module_batch_config
    beta = config["actor"]["beta"]
    action_dim = batch[SampleBatch.ACTIONS][0].numel()

    policy_out = module.actor.policy(batch[SampleBatch.CUR_OBS])
    norms = policy_out.norm(p=1, dim=-1, keepdim=True) / action_dim
    assert policy_out.shape[-1] == action_dim
    assert policy_out.dtype == torch.float32
    assert (norms <= (beta + torch.finfo(torch.float32).eps)).all()


def test_behavior(module_batch_config):
    module, batch, config = module_batch_config
    exploration = config["exploration"]
    exploration_gaussian_sigma = config["exploration_gaussian_sigma"]
    action = batch[SampleBatch.ACTIONS]

    samples = module.actor.behavior(batch[SampleBatch.CUR_OBS])
    samples_ = module.actor.behavior(batch[SampleBatch.CUR_OBS])
    assert samples.shape == action.shape
    assert samples.dtype == torch.float32
    assert not (
        (exploration == "gaussian" and exploration_gaussian_sigma != 0)
        and torch.allclose(samples, samples_)
    )
