import pytest
import torch
from ray.rllib import SampleBatch

from raylab.torch.nn.actor import (
    DeterministicPolicy,
    MLPContinuousPolicy,
    MLPDeterministicPolicy,
)
from raylab.torch.nn.critic import ActionValueCritic


@pytest.fixture(
    params=(pytest.param(True, marks=pytest.mark.slow), False),
    ids=("TorchScript", "Eager"),
    scope="module",
)
def torch_script(request):
    return request.param


@pytest.fixture(scope="module")
def batch(obs_space, action_space):
    from raylab.utils.debug import fake_batch

    samples = fake_batch(obs_space, action_space, batch_size=32)
    return {k: torch.from_numpy(v) for k, v in samples.items()}


@pytest.fixture
def obs(batch):
    return batch[SampleBatch.CUR_OBS]


@pytest.fixture
def action(batch):
    return batch[SampleBatch.ACTIONS]


@pytest.fixture
def deterministic_policies(obs_space, action_space):
    spec = MLPDeterministicPolicy.spec_cls(
        units=(32,), activation="ReLU", norm_beta=1.2
    )
    policy = MLPDeterministicPolicy(obs_space, action_space, spec)
    target_policy = DeterministicPolicy.add_gaussian_noise(policy, noise_stddev=0.3)
    return policy, target_policy


@pytest.fixture(params=(True, False), ids=(f"PiScaleDep({b})" for b in (True, False)))
def policy_input_scale(request):
    return request.param


@pytest.fixture
def stochastic_policy(obs_space, action_space, policy_input_scale):
    config = {"encoder": {"units": (32,)}}
    mlp_spec = MLPContinuousPolicy.spec_cls.from_dict(config)
    return MLPContinuousPolicy(
        obs_space, action_space, mlp_spec, input_dependent_scale=policy_input_scale
    )


@pytest.fixture(params=(1, 2), ids=(f"Critics({n})" for n in (1, 2)))
def action_critics(request, obs_space, action_space):
    config = {
        "encoder": {"units": [32]},
        "double_q": request.param == 2,
        "parallelize": False,
    }
    spec = ActionValueCritic.spec_cls.from_dict(config)

    act_critic = ActionValueCritic(obs_space, action_space, spec)
    return act_critic.q_values, act_critic.target_q_values
