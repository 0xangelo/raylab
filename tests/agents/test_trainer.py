# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import contextlib
import functools

import pytest
import ray
from ray.rllib import Policy
from ray.rllib.optimizers import PolicyOptimizer

from raylab.agents.trainer import StatsTracker
from raylab.agents.trainer import Trainer
from raylab.agents.trainer import with_common_config

from ..mock_env import MockEnv


class DummyPolicy(Policy):
    # pylint:disable=abstract-method,too-many-arguments
    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        explore=None,
        timestep=None,
        **kwargs,
    ):
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def get_weights(self):
        return []

    def set_weights(self, weights):
        pass


class MinimalTrainer(Trainer):
    _name = "MinimalTrainer"
    _default_config = with_common_config(
        {"workers": False, "optimizer": False, "tracker": False}
    )
    _policy = DummyPolicy

    def _init(self, config, env_creator):
        def make_workers():
            return self._make_workers(
                env_creator, self._policy, config, num_workers=config["num_workers"]
            )

        if config["tracker"]:
            self.tracker = StatsTracker(make_workers())
        elif config["optimizer"]:
            self.optimizer = PolicyOptimizer(make_workers())
        elif config["workers"]:
            self.workers = make_workers()

    def _train(self):
        return self._log_metrics({}, 0)


def setup_module():
    ray.init()


def teardown_module():
    ray.shutdown()


@pytest.fixture(params=(True, False), ids=(f"Tracker({b})" for b in (True, False)))
def tracker(request):
    return request.param


@pytest.fixture(params=(True, False), ids=(f"Workers({b})" for b in (True, False)))
def workers(request):
    return request.param


@pytest.fixture(params=(True, False), ids=(f"Optimizer({b})" for b in (True, False)))
def optimizer(request):
    return request.param


@pytest.fixture
def trainer_cls():
    return functools.partial(MinimalTrainer, env=MockEnv)


def test_trainer(trainer_cls, tracker, workers, optimizer):
    should_have_workers = any((tracker, workers, optimizer))
    context = (
        contextlib.nullcontext() if should_have_workers else pytest.warns(UserWarning)
    )
    with context:
        trainer = trainer_cls(
            config=dict(
                tracker=tracker, workers=workers, optimizer=optimizer, num_workers=0
            )
        )

    assert hasattr(trainer, "tracker")

    if should_have_workers:
        assert trainer.tracker.workers
        worker = trainer.tracker.workers.local_worker()
        _ = worker.sample()

        metrics = trainer.collect_metrics()
        assert isinstance(metrics, dict)


def test_evaluate_first(trainer_cls):
    trainer = trainer_cls(
        config=dict(
            workers=True,
            num_workers=0,
            evaluation_interval=10,
            evaluation_config=dict(batch_mode="truncate_episodes"),
            evaluation_num_episodes=1,
            evaluation_num_workers=0,
        )
    )
    assert hasattr(trainer, "evaluation_metrics")
    assert not trainer.evaluation_metrics

    res = trainer.train()
    assert "evaluation" in res
    assert hasattr(trainer, "evaluation_metrics")
    assert trainer.evaluation_metrics

    # Assert evaluation is not run again
    metrics = trainer.evaluation_metrics
    trainer.train()
    assert set(metrics.keys()) == set(trainer.evaluation_metrics.keys())
    assert all(metrics[k] == trainer.evaluation_metrics[k] for k in metrics.keys())
