# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
from ray.rllib import SampleBatch

from raylab.losses import ModelEnsembleMLE
from raylab.losses import SPAML
from raylab.losses.abstract import Loss


@pytest.fixture
def loss_fn(models, stochastic_policy, action_critics, reward_fn, termination_fn):
    critics, _ = action_critics
    loss_fn = SPAML(models, stochastic_policy, critics)
    loss_fn.set_reward_fn(reward_fn)
    loss_fn.set_termination_fn(termination_fn)
    return loss_fn


@pytest.fixture
def n_models(models):
    return len(models)


def test_init(loss_fn, n_models):
    assert isinstance(loss_fn, Loss)
    assert hasattr(loss_fn, "batch_keys")

    assert hasattr(loss_fn, "gamma")
    assert hasattr(loss_fn, "alpha")
    assert hasattr(loss_fn, "grad_estimator")
    assert hasattr(loss_fn, "lambda_")
    assert hasattr(loss_fn, "_modules")
    assert "models" in loss_fn._modules
    assert "policy" in loss_fn._modules
    assert "critics" in loss_fn._modules

    assert hasattr(loss_fn, "_loss_mle")
    assert isinstance(loss_fn._loss_mle, ModelEnsembleMLE)

    assert hasattr(loss_fn, "_env")
    assert loss_fn.initialized
    assert loss_fn.ensemble_size == n_models


def test_call(loss_fn, batch, n_models):
    tensor, info = loss_fn(batch)
    assert torch.is_tensor(tensor)
    assert len(tensor) == n_models

    assert isinstance(info, dict)
    assert all(isinstance(k, str) for k in info.keys())
    assert all(isinstance(v, (float, int)) for v in info.values())
    assert all(f"loss(models[{i}])" in info for i in range(n_models))
    assert "loss(daml)" in info
    assert "loss(mle)" in info


def test_compile(loss_fn, batch):
    loss_fn.compile()
    assert all(
        isinstance(loss_fn._modules[k], torch.jit.ScriptModule)
        for k in "models critics".split()
    )
    loss_fn(batch)


@pytest.fixture
def batch_shape(batch):
    return (len(batch[SampleBatch.CUR_OBS]),)


@pytest.fixture
def obs(n_models, batch_shape, obs_space):
    return torch.randn((n_models,) + batch_shape + obs_space.shape)


@pytest.fixture
def action(n_models, batch_shape, action_space):
    tensor = torch.randn((n_models,) + batch_shape + action_space.shape)
    return tensor.requires_grad_(True)


def test_expand_for_each_model(loss_fn, batch, obs):
    batch_obs = batch[SampleBatch.CUR_OBS]
    res = loss_fn.expand_for_each_model(batch_obs)
    assert torch.is_tensor(res)
    assert res.shape == obs.shape


def test_generate_action(loss_fn, obs, action):
    res = loss_fn.generate_action(obs)
    assert torch.is_tensor(res)
    assert res.shape == action.shape
    assert res.requires_grad


def test_zero_step_action_value(loss_fn, obs, action, n_models, batch_shape):
    value = loss_fn.zero_step_action_value(obs, action)
    assert torch.is_tensor(value)
    assert value.shape == (n_models,) + batch_shape


@pytest.fixture(params="SF PD".split())
def grad_estimator(request):
    return request.param


def test_one_step_action_value_surrogate(
    loss_fn, obs, action, grad_estimator, n_models, batch_shape
):
    # pylint:disable=too-many-arguments
    loss_fn.grad_estimator = grad_estimator
    value = loss_fn.one_step_action_value_surrogate(obs, action)
    assert torch.is_tensor(value)
    assert value.shape == (n_models,) + batch_shape


def test_transition(loss_fn, obs, action, n_models, batch_shape):
    next_obs, log_prob = loss_fn.transition(obs, action)
    assert torch.is_tensor(next_obs)
    assert torch.is_tensor(log_prob)
    assert next_obs.shape == obs.shape
    assert log_prob.shape == (n_models,) + batch_shape


def test_action_gradient_loss(loss_fn, obs, action, n_models):
    value_target = loss_fn.zero_step_action_value(obs, action)
    value_pred = loss_fn.one_step_action_value_surrogate(obs, action)
    loss = loss_fn.action_gradient_loss(action, value_target, value_pred)

    assert torch.is_tensor(loss)
    assert loss.shape == (n_models,)


def test_maximum_likelihood_loss(loss_fn, batch, n_models):
    loss = loss_fn.maximum_likelihood_loss(batch)

    assert torch.is_tensor(loss)
    assert loss.shape == (n_models,)
