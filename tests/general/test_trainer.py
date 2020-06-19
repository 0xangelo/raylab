# pylint: disable=missing-docstring,redefined-outer-name,protected-access
from collections import defaultdict

import pytest


CONFIG = defaultdict(
    lambda: {
        "num_workers": 0,
        "timesteps_per_iteration": 1,
        "rollout_fragment_length": 25,
        "batch_mode": "truncate_episodes",
        "evaluation_config": {"explore": False},
        "evaluation_interval": 1,
        "evaluation_num_episodes": 1,
        "evaluation_num_workers": 0,
    }
)
CONFIG["MBPO"].update(
    {"model_rollouts": 1, "policy_improvements": 1, "model_training": {"max_epochs": 1}}
)
CONFIG["MAGE"].update({"policy_improvements": 1, "model_training": {"max_epochs": 1}})


@pytest.fixture(
    scope="module",
    params=(True, False),
    ids=(f"CompilePolicy({b})" for b in (True, False)),
)
def compile_policy(request):
    return request.param


@pytest.fixture(scope="module")
def trainer(trainer_cls, compile_policy):
    config = CONFIG[trainer_cls._name].copy()
    config["compile_policy"] = compile_policy
    return trainer_cls(env="MockEnv", config=config)


@pytest.mark.slow
def test_trainer_step(trainer):
    info = trainer.train()
    assert isinstance(info, dict)
    assert all(isinstance(k, str) for k in info)


def test_trainer_eval(trainer):
    metrics = trainer._evaluate()
    assert isinstance(metrics, dict)


def test_trainer_restore(trainer):
    obj = trainer.save_to_object()
    trainer.restore_from_object(obj)
