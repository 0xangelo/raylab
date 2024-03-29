from __future__ import annotations

from collections import defaultdict
from typing import Callable

import pytest
from ray.rllib import RolloutWorker, SampleBatch
from ray.tune.logger import Logger

from raylab.agents.registry import AGENTS
from raylab.envs import get_env_creator

TRAINER_NAMES, TRAINER_IMPORTS = zip(*AGENTS.items())

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


@pytest.fixture(scope="module", params=TRAINER_IMPORTS, ids=TRAINER_NAMES)
def trainer_cls(request):
    return request.param()


@pytest.fixture(
    scope="module", params=(True, False), ids=lambda b: f"CompilePolicy:{b}"
)
def compile_policy(request):
    return request.param


@pytest.fixture(scope="module")
def trainer(trainer_cls, compile_policy, logger_creator: Callable[[dict], Logger]):
    name = trainer_cls._name
    defaults = trainer_cls.options.defaults
    config = CONFIG[name].copy()

    if "learning_starts" in defaults:
        config["learning_starts"] = 1
    if name == "SVG(inf)":
        config["batch_mode"] = "complete_episodes"

    config["policy"] = {"compile": compile_policy}

    return trainer_cls(env="MockEnv", config=config, logger_creator=logger_creator)


@pytest.fixture
def policy_cls(trainer):
    return trainer.get_policy_class()


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
def worker(env_name, policy_cls, worker_kwargs):
    return RolloutWorker(
        env_creator=get_env_creator(env_name),
        policy_spec=policy_cls,
        policy_config={"env": env_name, "framework": "torch"},
        **worker_kwargs,
    )


@pytest.mark.filterwarnings("ignore:.+ is incompatible with TorchScript::raylab")
def test_compute_single_action(env_, env_name, policy_cls):
    env = env_
    policy = policy_cls(env.observation_space, env.action_space, {"env": env_name})

    obs = env.observation_space.sample()
    action, states, info = policy.compute_single_action(obs, [])
    assert action in env.action_space
    assert isinstance(states, list)
    assert isinstance(info, dict)


def test_policy_in_rollout_worker(worker):
    policy = worker.get_policy()
    _, _, extra_fetches = policy.compute_single_action(
        policy.observation_space.sample()
    )
    traj = worker.sample()
    assert isinstance(traj, SampleBatch)
    assert all(k in traj for k in extra_fetches)
