# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
import torch.nn as nn

import raylab.pytorch.nn as nnx
import raylab.pytorch.nn.distributions as ptd
from raylab.modules.mixins.action_value_mixin import ActionValueFunction
from raylab.modules.mixins.deterministic_actor_mixin import DeterministicPolicy
from raylab.modules.mixins.stochastic_actor_mixin import StochasticPolicy
from raylab.modules.mixins.stochastic_model_mixin import StochasticModelMixin
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
def make_model(obs_space, action_space):
    config = {
        "encoder": {"units": (32,)},
        "residual": True,
        "input_dependent_scale": True,
    }

    def factory():
        return StochasticModelMixin.build_single_model(obs_space, action_space, config)

    return factory


@pytest.fixture(params=(1, 2, 4), ids=(f"Models({n})" for n in (1, 2, 4)))
def models(request, make_model):
    return nn.ModuleList([make_model() for _ in range(request.param)])


@pytest.fixture(params=(1, 2), ids=(f"Critics({n})" for n in (1, 2)))
def action_critics(request, obs_space, action_space):
    def critic():
        return ActionValueFunction.from_scratch(
            obs_space.shape[0], action_space.shape[0], units=(32,)
        )

    critics = nn.ModuleList([critic() for _ in range(request.param)])
    target_critics = nn.ModuleList([critic() for _ in range(request.param)])
    target_critics.load_state_dict(critics.state_dict())
    return critics, target_critics


@pytest.fixture
def deterministic_policies(obs_space, action_space):
    policy = DeterministicPolicy.from_scratch(
        obs_space, action_space, {"beta": 1.2, "encoder": {"units": (32,)}}
    )
    target_policy = DeterministicPolicy.from_existing(policy, noise=0.3)
    return policy, target_policy


@pytest.fixture(params=(True, False), ids=(f"PiScaleDep({b})" for b in (True, False)))
def policy_input_scale(request):
    return request.param


@pytest.fixture
def stochastic_policy(obs_space, action_space, policy_input_scale):
    config = {
        "encoder": {"units": (32,)},
        "input_dependent_scale": policy_input_scale,
    }

    logits = nnx.FullyConnected(in_features=obs_space.shape[0], **config["encoder"])
    params = nnx.NormalParams(
        logits.out_features,
        action_space.shape[0],
        input_dependent_scale=config["input_dependent_scale"],
    )
    params_module = nn.Sequential(logits, params)
    dist_module = ptd.TransformedDistribution(
        ptd.Independent(ptd.Normal(), reinterpreted_batch_ndims=1),
        ptd.flows.TanhSquashTransform(
            low=torch.as_tensor(action_space.low),
            high=torch.as_tensor(action_space.high),
            event_dim=1,
        ),
    )
    return StochasticPolicy(params_module, dist_module)
