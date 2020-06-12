# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
import torch.nn as nn

from raylab.losses import MAGE
from raylab.modules.mixins.action_value_mixin import ActionValueFunction
from raylab.modules.mixins.deterministic_actor_mixin import DeterministicPolicy
from raylab.modules.mixins.stochastic_model_mixin import StochasticModelMixin
from raylab.utils.debug import fake_batch


@pytest.fixture(params=(1, 2), ids=(f"Critics({n})" for n in (1, 2)))
def critics(request, obs_space, action_space):
    def critic():
        return ActionValueFunction.from_scratch(
            obs_space.shape[0], action_space.shape[0], units=(32,)
        )

    critics = nn.ModuleList([critic() for _ in range(request.param)])
    target_critics = nn.ModuleList([critic() for _ in range(request.param)])
    target_critics.load_state_dict(critics.state_dict())
    return critics, target_critics


@pytest.fixture
def policy(obs_space, action_space):
    return DeterministicPolicy.from_scratch(
        obs_space, action_space, {"beta": 1.2, "encoder": {"units": (32,)}}
    )


@pytest.fixture(
    params=(True, False), ids=(f"ResidualModel({b})" for b in (True, False))
)
def residual(request):
    return request.param


@pytest.fixture(params=(1, 2, 4), ids=(f"Models({n})" for n in (1, 2, 4)))
def models(request, residual, obs_space, action_space):
    config = {
        "encoder": {"units": (32,)},
        "residual": residual,
        "input_dependent_scale": True,
    }

    def model():
        return StochasticModelMixin.build_single_model(obs_space, action_space, config)

    return nn.ModuleList([model() for _ in range(request.param)])


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
def modules(critics, policy, models):
    critics, target_critics = critics
    return critics, target_critics, policy, models


@pytest.fixture
def loss_fn(modules, reward_fn, termination_fn):
    critics, target_critics, policy, models = modules

    loss_fn = MAGE(critics, target_critics, policy, models)
    loss_fn.set_reward_fn(reward_fn)
    loss_fn.set_termination_fn(termination_fn)
    return loss_fn


@pytest.fixture(scope="module")
def batch(obs_space, action_space):
    samples = fake_batch(obs_space, action_space, batch_size=256)
    return {k: torch.from_numpy(v) for k, v in samples.items()}


def test_mage_init(modules, reward_fn, termination_fn):
    critics, target_critics, policy, models = modules

    loss_fn = MAGE(critics, target_critics, policy, models)
    assert not loss_fn.initialized
    assert hasattr(loss_fn, "gamma")
    assert hasattr(loss_fn, "lambda_")
    assert hasattr(loss_fn, "_modules")
    assert "critics" in loss_fn._modules
    assert "target_critics" in loss_fn._modules
    assert "policy" in loss_fn._modules
    assert "models" in loss_fn._modules
    assert hasattr(loss_fn, "_rng")

    loss_fn.set_reward_fn(reward_fn)
    loss_fn.set_termination_fn(termination_fn)
    assert loss_fn.initialized

    loss_fn.seed(42)
    assert hasattr(loss_fn, "_rng")


def test_mage_call(loss_fn, batch):
    loss, info = loss_fn(batch)

    assert torch.is_tensor(loss)
    assert isinstance(info, dict)
    assert all(isinstance(k, str) for k in info.keys())
    assert all(isinstance(v, float) for v in info.values())


def test_gradient_is_finite(loss_fn, batch):
    loss, _ = loss_fn(batch)

    loss.backward()
    critics = loss_fn._modules["critics"]
    assert all(p.grad is not None for p in critics.parameters())
    assert all(torch.isfinite(p.grad).all() for p in critics.parameters())


def test_rng(loss_fn):
    models = loss_fn._modules["models"]
    loss_fn.seed(42)
    model = loss_fn._rng.choice(models)
    id_ = id(model)
    loss_fn.seed(42)
    model = loss_fn._rng.choice(models)
    assert id_ == id(model)


def test_grad_loss_gradient_propagation(loss_fn, batch):
    obs, action, next_obs = loss_fn.generate_transition(batch)
    delta = loss_fn.temporal_diff_error(obs, action, next_obs)
    _ = loss_fn.gradient_loss(delta, action)

    parameters = set(loss_fn._modules.parameters())
    assert all(p.grad is None for p in parameters)
    assert action.grad_fn is not None
