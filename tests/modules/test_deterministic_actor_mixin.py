# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
import torch.nn as nn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import merge_dicts

from raylab.modules.deterministic_actor_mixin import DeterministicActorMixin


BASE_CONFIG = {
    "torch_script": False,
    "smooth_target_policy": False,
    "target_gaussian_sigma": 0.3,
    "perturbed_policy": False,
    "actor": {
        "units": (32, 32),
        "activation": "ReLU",
        "initializer_options": {"name": "xavier_uniform"},
        "layer_norm": False,
        "beta": 1.2,
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


BETA = (0.8, 1.2)


@pytest.fixture(scope="module", params=BETA, ids=(f"Beta{b}" for b in BETA))
def beta(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=(True, False),
    ids=("SmoothTargetPolicy", "HardTargetPolicy"),
)
def smooth_target_policy(request):
    return request.param


@pytest.fixture(
    scope="module", params=(True, False), ids=("PerturbedPolicy", "FixedPolicy")
)
def perturbed_policy(request):
    return request.param


@pytest.fixture(scope="module")
def full_config(beta, smooth_target_policy, perturbed_policy):
    return {
        "smooth_target_policy": smooth_target_policy,
        "actor": {"beta": beta},
        "perturbed_policy": perturbed_policy,
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
    module, batch, _ = module_batch_config
    action = batch[SampleBatch.ACTIONS]

    samples = module.actor.behavior(batch[SampleBatch.CUR_OBS])
    samples_ = module.actor.behavior(batch[SampleBatch.CUR_OBS])
    assert samples.shape == action.shape
    assert samples.dtype == torch.float32
    assert torch.allclose(samples, samples_)
