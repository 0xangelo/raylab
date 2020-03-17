# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest

from raylab.modules.deterministic_actor_critic import DeterministicActorCritic


@pytest.fixture(params=(True, False), ids=("double_q", "single_q"))
def double_q(request):
    return request.param


@pytest.fixture(params=(None, "gaussian", "parameter_noise"))
def exploration(request):
    return request.param


@pytest.fixture(
    params=(True, False), ids=("smooth_target_policy", "hard_target_policy")
)
def target_policy_smoothing(request):
    return request.param


@pytest.fixture
def config(double_q, exploration, target_policy_smoothing):
    return {
        "clipped_double_q": double_q,
        "exploration": exploration,
        "exploration_gaussian_sigma": 0.3,
        "target_policy_smoothing": target_policy_smoothing,
        "target_gaussian_sigma": 0.3,
        "policy": {
            "units": (400, 300),
            "activation": "ReLU",
            "initializer_options": {"name": "xavier_uniform"},
            # === SQUASHING EXPLORATION PROBLEM ===
            # Maximum l1 norm of the policy's output vector before the squashing
            # function
            "beta": 1.2,
        },
        "critic": {
            "units": (400, 300),
            "activation": "ReLU",
            "initializer_options": {"name": "xavier_uniform"},
            "delay_action": True,
        },
    }


def test_module_creation(obs_space, action_space, config):
    module = DeterministicActorCritic(obs_space, action_space, config)

    assert "policy" in module
    assert "critics" in module
    assert "target_critics" in module
    expected_n_critics = 2 if config["clipped_double_q"] else 1
    assert len(module.critics) == expected_n_critics
