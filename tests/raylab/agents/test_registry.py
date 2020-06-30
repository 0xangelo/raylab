# pylint:disable=missing-docstring,redefined-outer-name,protected-access
from collections import defaultdict

import pytest
from ray.rllib import RolloutWorker
from ray.rllib import SampleBatch

from raylab.agents.registry import AGENTS

TRAINER_NAMES, TRAINER_IMPORTS = zip(*AGENTS.items())


@pytest.fixture(scope="module", params=TRAINER_IMPORTS, ids=TRAINER_NAMES)
def trainer_cls(request):
    return request.param()


@pytest.fixture
def policy_cls(trainer_cls):
    return trainer_cls._policy


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
CONFIG["MAPO"].update({"policy_improvements": 1, "model_training": {"max_epochs": 1}})


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


@pytest.fixture
def worker_kwargs():
    return {"rollout_fragment_length": 200, "batch_mode": "truncate_episodes"}


@pytest.fixture
def worker(envs, env_name, policy_cls, worker_kwargs):
    return RolloutWorker(
        env_creator=envs[env_name],
        policy=policy_cls,
        policy_config={"env": env_name},
        **worker_kwargs,
    )


def test_compute_single_action(envs, env_name, policy_cls):
    env = envs[env_name]({})
    policy = policy_cls(env.observation_space, env.action_space, {"env": env_name})

    obs = env.observation_space.sample()
    action, states, info = policy.compute_single_action(obs, [])
    assert action in env.action_space
    assert isinstance(states, list)
    assert isinstance(info, dict)


def test_policy_in_rollout_worker(worker):
    traj = worker.sample()
    assert isinstance(traj, SampleBatch)
