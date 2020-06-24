# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch

from raylab.losses.mage import MAGE
from raylab.losses.mage import MAGEModules


@pytest.fixture
def modules(action_critics, deterministic_policies, models):
    critics, target_critics = action_critics
    policy, target_policy = deterministic_policies
    return MAGEModules(
        critics=critics,
        target_critics=target_critics,
        policy=policy,
        target_policy=target_policy,
        models=models,
    )


@pytest.fixture
def loss_fn(modules, reward_fn, termination_fn):
    loss_fn = MAGE(modules)
    loss_fn.set_reward_fn(reward_fn)
    loss_fn.set_termination_fn(termination_fn)
    return loss_fn


def test_mage_init(modules, reward_fn, termination_fn):
    loss_fn = MAGE(modules)
    assert not loss_fn.initialized
    assert hasattr(loss_fn, "gamma")
    assert hasattr(loss_fn, "lambda_")
    assert hasattr(loss_fn, "_modules")
    assert "critics" in loss_fn._modules
    assert "target_critics" in loss_fn._modules
    assert "policy" in loss_fn._modules
    assert "target_policy" in loss_fn._modules
    assert "models" in loss_fn._modules
    assert hasattr(loss_fn, "_rng")

    loss_fn.set_reward_fn(reward_fn)
    loss_fn.set_termination_fn(termination_fn)
    assert loss_fn.initialized

    loss_fn.seed(42)
    assert hasattr(loss_fn, "_rng")


@pytest.fixture(params=(False, True), ids=("Eager", "Script"))
def script(request):
    return request.param


def test_compile(loss_fn):
    loss_fn.compile()
    assert all(isinstance(v, torch.jit.ScriptModule) for v in loss_fn._modules.values())


def test_mage_call(loss_fn, batch, script):
    if script:
        loss_fn.compile()
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
