# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import contextlib
import functools

import pytest
from ray.rllib import Policy
from ray.rllib.optimizers import PolicyOptimizer

from raylab.agents.trainer import config
from raylab.agents.trainer import Trainer


@pytest.fixture(scope="module")
def policy_cls():
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

    return DummyPolicy


@pytest.fixture(scope="module")
def trainer_cls(policy_cls):
    @config("workers", False)
    @config("optim", False)
    @config(
        "arbitrary",
        {"type": "one", "key": "value"},
        allow_unknown_subkeys=True,
        override_all_if_type_changes=True,
    )
    class MinimalTrainer(Trainer):
        _name = "MinimalTrainer"
        _policy = policy_cls

        def _init(self, config, env_creator):
            def make_workers():
                return self._make_workers(
                    env_creator, self._policy, config, num_workers=config["num_workers"]
                )

            if config["optim"]:
                self.optimizer = PolicyOptimizer(make_workers())
            elif config["workers"]:
                self.workers = make_workers()

        def _train(self):
            return self._log_metrics({}, 0)

    return functools.partial(MinimalTrainer, env="MockEnv")


@pytest.fixture(params=(True, False), ids=(f"Workers({b})" for b in (True, False)))
def workers(request):
    return request.param


@pytest.fixture(params=(True, False), ids=(f"Optimizer({b})" for b in (True, False)))
def optim(request):
    return request.param


def test_trainer(trainer_cls, workers, optim):
    should_have_workers = any((workers, optim))
    context = (
        contextlib.nullcontext() if should_have_workers else pytest.warns(UserWarning)
    )
    with context:
        trainer = trainer_cls(config=dict(workers=workers, optim=optim, num_workers=0))

    assert not should_have_workers or hasattr(trainer, "metrics")

    if should_have_workers:
        assert trainer.metrics.workers
        worker = trainer.metrics.workers.local_worker()
        _ = worker.sample()

        metrics = trainer.collect_metrics()
        assert isinstance(metrics, dict)
        assert "episode_reward_mean" in metrics
        assert "episode_reward_min" in metrics
        assert "episode_reward_max" in metrics
        assert "episode_len_mean" in metrics

    trainer.stop()


@pytest.fixture(
    params=({}, {"type": "two", "param": "default"}),
    ids="Unchanged OverrideType".split(),
)
def arbitrary(request):
    return request.param


def test_override_all_if_type_changes(trainer_cls, arbitrary):
    trainer = trainer_cls(config=dict(arbitrary=arbitrary))

    subconfig = trainer.config["arbitrary"]
    if arbitrary:
        assert "key" not in subconfig
        assert "param" in subconfig
        assert subconfig["param"] == "default"
    else:
        assert "key" in subconfig
        assert subconfig["key"] == "value"


@pytest.fixture
def eval_trainer(trainer_cls):
    return trainer_cls(
        config=dict(
            workers=True,
            num_workers=0,
            evaluation_interval=10,
            evaluation_config=dict(batch_mode="truncate_episodes"),
            evaluation_num_episodes=1,
            evaluation_num_workers=0,
        )
    )


def test_evaluate_first(eval_trainer):
    trainer = eval_trainer
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
