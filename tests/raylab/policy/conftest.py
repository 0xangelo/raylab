import pytest
import torch
from nnrl.nn.actor import (
    Alpha,
    DeterministicPolicy,
    MLPContinuousPolicy,
    MLPDeterministicPolicy,
)
from nnrl.nn.critic import ActionValueCritic, MLPVValue
from nnrl.nn.model import EnsembleSpec, build_ensemble, build_single
from ray.rllib import SampleBatch

from raylab.utils.debug import fake_batch


@pytest.fixture(scope="module")
def reward_fn():
    def func(obs, act, new_obs):
        return new_obs[..., 0] - obs[..., 0] - act.norm(dim=-1)

    return func


@pytest.fixture(scope="module")
def termination_fn():
    def func(obs, *_):
        return torch.randn_like(obs[..., 0]) > 0

    return func


@pytest.fixture
def batch(obs_space, action_space):
    samples = fake_batch(obs_space, action_space, batch_size=256)
    return {k: torch.from_numpy(v) for k, v in samples.items()}


@pytest.fixture
def obs(batch):
    return batch[SampleBatch.CUR_OBS]


@pytest.fixture
def rew(batch):
    return batch[SampleBatch.REWARDS]


@pytest.fixture
def done(batch):
    return batch[SampleBatch.DONES]


@pytest.fixture
def new_obs(batch):
    return batch[SampleBatch.NEXT_OBS]


@pytest.fixture
def model_spec():
    spec = EnsembleSpec()
    spec.network.units = (32,)
    spec.network.input_dependent_scale = True
    spec.residual = True
    return spec


@pytest.fixture
def model(obs_space, action_space, model_spec):
    return build_single(obs_space, action_space, model_spec)


@pytest.fixture(params=(1, 2, 4), ids=(f"Models({n})" for n in (1, 2, 4)))
def models(request, obs_space, action_space, model_spec):
    spec = model_spec
    spec.ensemble_size = request.param
    spec.parallelize = True

    return build_ensemble(obs_space, action_space, spec)


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


@pytest.fixture
def state_critics(obs_space):
    spec = MLPVValue.spec_cls()
    spec.units = (32,)
    spec.activation = "ReLU"
    spec.layer_norm = False

    main, target = MLPVValue(obs_space, spec), MLPVValue(obs_space, spec)
    return main, target


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


@pytest.fixture
def alpha_module():
    return Alpha(1.0)
