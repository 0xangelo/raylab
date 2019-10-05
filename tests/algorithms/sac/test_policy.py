import pytest
import torch
from ray.rllib.policy.sample_batch import SampleBatch


@pytest.fixture(params=(True, False))
def input_dependent_scale(request):
    return {"module": {"policy": {"input_dependent_scale": request.param}}}


@pytest.fixture(params=(True, False))
def clipped_double_q(request):
    return {"clipped_double_q": request.param}


def test_policy_output(policy_and_batch_fn, input_dependent_scale):
    policy, batch = policy_and_batch_fn(input_dependent_scale)

    params = policy.module.policy(batch[SampleBatch.CUR_OBS])
    assert "loc" in params
    assert "scale_diag" in params

    loc, scale_diag = [params[k] for k in ("loc", "scale_diag")]
    assert loc.shape == (10,) + policy.action_space.shape
    assert scale_diag.shape == (10,) + policy.action_space.shape
    assert loc.dtype == torch.float32
    assert scale_diag.dtype == torch.float32

    pi_params = set(policy.module.policy.parameters())
    for par in pi_params:
        par.grad = None
    loc.mean().backward()
    assert any(p.grad is not None for p in pi_params)
    assert all(p.grad is None for p in set(policy.module.parameters()) - pi_params)

    for par in pi_params:
        par.grad = None
    policy.module.policy(batch[SampleBatch.CUR_OBS])["scale_diag"].mean().backward()
    assert any(p.grad is not None for p in pi_params)
    assert all(p.grad is None for p in set(policy.module.parameters()) - pi_params)


def test_policy_policy_loss(
    policy_and_batch_fn, clipped_double_q, input_dependent_scale
):
    policy, batch = policy_and_batch_fn({**clipped_double_q, **input_dependent_scale})
    loss = policy.compute_policy_loss(batch, policy.module, policy.config)

    assert loss.shape == ()
    assert loss.dtype == torch.float32

    loss.backward()
    assert all(p.grad is not None for p in policy.module.policy.parameters())
    assert policy.module.log_alpha.grad is not None
    assert all(p.grad is not None for p in policy.module.critic.parameters())
    if policy.config["clipped_double_q"]:
        assert all(p.grad is not None for p in policy.module.twin_critic.parameters())
