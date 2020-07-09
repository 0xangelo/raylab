import pytest
import torch
from ray.rllib import SampleBatch


@pytest.fixture(scope="module")
def module_cls():
    from raylab.policy.modules.critic.action_value import ActionValueCritic

    return ActionValueCritic


@pytest.fixture(params=(True, False), ids="DoubleQ SingleQ".split())
def double_q(request):
    return request.param


@pytest.fixture(params=(True, False), ids=lambda x: "Parallelize({x})")
def parallelize(request):
    return request.param


@pytest.fixture
def spec(module_cls, double_q, parallelize):
    return module_cls.spec_cls(double_q=double_q, parallelize=parallelize)


@pytest.fixture
def module(module_cls, obs_space, action_space, spec):
    return module_cls(obs_space, action_space, spec)


def test_module_creation(module, batch, spec):
    double_q = spec.double_q

    for attr in "q_values target_q_values".split():
        assert hasattr(module, attr)
    expected_n_critics = 2 if double_q else 1
    assert len(module.q_values) == expected_n_critics

    q_values, targets = module.q_values, module.target_q_values
    vals = [
        m(batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS])
        for ensemble in (q_values, targets)
        for m in ensemble
    ]
    for val in vals:
        assert val.shape[-1] == 1
        assert val.dtype == torch.float32

    assert all(
        torch.allclose(p, t)
        for p, t in zip(q_values.parameters(), targets.parameters())
    )


def test_script(module):
    torch.jit.script(module)
