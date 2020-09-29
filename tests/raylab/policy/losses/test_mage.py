import pytest
import torch
from ray.rllib import SampleBatch

from raylab.policy.losses.mage import MAGE
from raylab.policy.modules.critic import HardValue


@pytest.fixture
def critics(action_critics):
    critics, _ = action_critics
    return critics


@pytest.fixture
def policy(deterministic_policies):
    policy, _ = deterministic_policies
    return policy


@pytest.fixture
def target_critic(action_critics, deterministic_policies):
    _, target_critics = action_critics
    _, target_policy = deterministic_policies
    return HardValue(target_policy, target_critics)


@pytest.fixture
def raw_loss(critics, policy, target_critic, models):
    return MAGE(critics, policy, target_critic, models)


@pytest.fixture
def loss_fn(raw_loss, reward_fn, termination_fn):
    # pylint:disable=too-many-arguments
    loss_fn = raw_loss
    loss_fn.set_reward_fn(reward_fn)
    loss_fn.set_termination_fn(termination_fn)
    return loss_fn


def test_mage_init(raw_loss, reward_fn, termination_fn):
    loss_fn = raw_loss
    assert hasattr(loss_fn, "gamma")
    assert hasattr(loss_fn, "lambd")
    assert hasattr(loss_fn, "critics")
    assert hasattr(loss_fn, "policy")
    assert hasattr(loss_fn, "target_critic")
    assert hasattr(loss_fn, "models")
    assert hasattr(loss_fn, "_rng")

    loss_fn.set_reward_fn(reward_fn)
    loss_fn.set_termination_fn(termination_fn)

    loss_fn.seed(42)
    assert hasattr(loss_fn, "_rng")


@pytest.fixture(params=(False, True), ids=("Eager", "Script"))
def script(request):
    return request.param


def test_compile(loss_fn):
    loss_fn.compile()
    assert not any(
        [
            isinstance(getattr(loss_fn, a), torch.jit.ScriptModule)
            for a in "critics policy target_critic models".split()
        ]
    )


def test_mage_call(loss_fn, batch, critics):
    loss, info = loss_fn(batch)

    assert torch.is_tensor(loss)
    assert isinstance(info, dict)
    assert all(isinstance(k, str) for k in info.keys())
    assert all(isinstance(v, float) for v in info.values())

    loss.backward()
    assert all([p.grad is not None for p in critics.parameters()])


@pytest.mark.skip(reason="https://github.com/pytorch/pytorch/issues/42459")
def test_script_backprop(loss_fn, batch, critics):
    loss_fn.compile()
    loss, _ = loss_fn(batch)

    loss.backward()
    assert all([p.grad is not None for p in critics.parameters()])


def test_gradient_is_finite(loss_fn, batch):
    loss, _ = loss_fn(batch)

    loss.backward()
    critics = loss_fn.critics
    assert all(p.grad is not None for p in critics.parameters())
    assert all(torch.isfinite(p.grad).all() for p in critics.parameters())


def test_rng(loss_fn):
    models = loss_fn.models
    loss_fn.seed(42)
    model = loss_fn._rng.choice(models)
    id_ = id(model)
    loss_fn.seed(42)
    model = loss_fn._rng.choice(models)
    assert id_ == id(model)


@pytest.fixture
def obs(batch):
    return batch[SampleBatch.CUR_OBS]


@pytest.fixture
def action(batch):
    return batch[SampleBatch.ACTIONS]


@pytest.fixture
def next_obs(batch):
    return batch[SampleBatch.NEXT_OBS]


@pytest.fixture
def rew(batch):
    return batch[SampleBatch.REWARDS]


def test_transition(loss_fn, obs, action):
    next_obs, dist_params = loss_fn.transition(obs, action)
    assert torch.is_tensor(next_obs)
    assert isinstance(dist_params, dict)
    assert all(
        [isinstance(k, str) and torch.is_tensor(v) for k, v in dist_params.items()]
    )


def test_delta(loss_fn, critics, obs, action, next_obs, rew):
    # pylint:disable=too-many-arguments
    diff = loss_fn.temporal_diff_error(obs, action, next_obs)
    assert torch.is_tensor(diff)
    assert diff.shape == rew.shape + (len(critics),)

    if len(critics) > 1:
        assert not torch.allclose(diff[..., 0], diff[..., 1], atol=1e-6)


def test_grad_loss_gradient_propagation(loss_fn, obs, action):
    action.requires_grad_(True)
    next_obs, _ = loss_fn.transition(obs, action)

    delta = loss_fn.temporal_diff_error(obs, action, next_obs)
    _ = loss_fn.gradient_loss(delta, action)

    parameters = set.union(
        *(
            set(getattr(loss_fn, a).parameters())
            for a in "critics policy target_critic models".split()
        )
    )
    assert all(p.grad is None for p in parameters)
