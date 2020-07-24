import contextlib
import itertools

import pytest
from ray.rllib import Policy
from ray.rllib.agents.trainer import Trainer as RLlibTrainer
from ray.rllib.optimizers import PolicyOptimizer

from raylab.agents.trainer import configure
from raylab.agents.trainer import option
from raylab.agents.trainer import Trainer


@pytest.fixture(scope="module")
def policy_cls():
    class DummyPolicy(Policy):
        # pylint:disable=abstract-method,too-many-arguments
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.param = 0
            self.param_seq = itertools.count()
            next(self.param_seq)

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

        def learn_on_batch(self, _):
            self.param = next(self.param_seq)
            return {"improved": True}

        def get_weights(self):
            return {"param": self.param}

        def set_weights(self, weights):
            self.param = weights["param"]

    return DummyPolicy


def test_dummy_policy(policy_cls, obs_space, action_space):
    policy = policy_cls(obs_space, action_space, {})
    assert policy.param == 0
    info = policy.learn_on_batch(None)
    assert "improved" in info
    assert info["improved"]
    assert policy.param == 1
    weights = policy.get_weights()
    assert "param" in weights
    assert weights["param"] == 1


@pytest.fixture(scope="module")
def trainer_cls(policy_cls):
    @configure
    @option("workers", False)
    @option("optim", False)
    @option("arbitrary/", allow_unknown_subkeys=True, override_all_if_type_changes=True)
    @option("arbitrary/type", "one")
    @option("arbitrary/key", "value")
    class MinimalTrainer(Trainer):
        _name = "MinimalTrainer"
        _policy = policy_cls

        def __init__(self, *args, **kwargs):
            super().__init__(*args, env="MockEnv", **kwargs)

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
            info = {}
            if hasattr(self, "workers"):
                policy = self.get_policy()
                info.update(policy.learn_on_batch(None))

            return info

    return MinimalTrainer


@pytest.fixture(params=(True, False), ids=(f"Workers({b})" for b in (True, False)))
def workers(request):
    return request.param


@pytest.fixture(params=(True, False), ids=(f"Optimizer({b})" for b in (True, False)))
def optim(request):
    return request.param


def test_default_config(trainer_cls):
    assert "workers" in trainer_cls.options.defaults
    assert "optim" in trainer_cls.options.defaults
    assert "arbitrary" in trainer_cls.options.defaults
    assert "arbitrary" in trainer_cls.options.allow_unknown_subkeys
    assert "arbitrary" in trainer_cls.options.override_all_if_type_changes


def test_metrics_creation(trainer_cls, workers, optim):
    should_have_workers = any((workers, optim))
    context = (
        contextlib.nullcontext() if should_have_workers else pytest.warns(UserWarning)
    )
    with context:
        trainer = trainer_cls(config=dict(workers=workers, optim=optim, num_workers=0))

    assert not should_have_workers or hasattr(trainer, "metrics")


@pytest.fixture(scope="module")
def trainable_info_keys():
    """Keys returned on any call to a subclass of `ray.tune.Trainable`."""
    return {
        "experiment_id",
        "date",
        "timestamp",
        "time_this_iter_s",
        "time_total_s",
        "pid",
        "hostname",
        "node_ip",
        "config",
        "time_since_restore",
        "timesteps_since_restore",
        "iterations_since_restore",
    }


@pytest.fixture
def trainer(trainer_cls, workers, optim):
    return trainer_cls(config=dict(workers=workers, optim=optim, num_workers=0))


def test_has_optimizer_and_worker(trainer, workers, optim):
    should_have_optimizer_and_worker = any([workers, optim])
    assert not should_have_optimizer_and_worker or trainer._has_policy_optimizer()
    assert not should_have_optimizer_and_worker or hasattr(trainer, "workers")


def test_train(trainer, workers, optim, trainable_info_keys):
    should_learn = any((workers, optim))
    if should_learn:
        info = trainer.train()
        info_keys = set(info.keys())
        assert all(key in info_keys for key in trainable_info_keys)

        info_keys.difference_update(trainable_info_keys)
        assert "improved" in info_keys
        assert info["improved"] is True

        assert trainer.get_policy().param == 1
        checkpoint = trainer.save_to_object()
        trainer.train()
        assert trainer.get_policy().param == 2

        trainer.restore_from_object(checkpoint)
        weights = trainer.get_policy().get_weights()
        assert "param" in weights
        assert weights["param"] == 1


def test_returns_metrics(trainer_cls, workers, optim):
    should_have_workers = any((workers, optim))
    trainer = trainer_cls(config=dict(workers=workers, optim=optim, num_workers=0))

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


def test_preserve_original_trainer_attr(trainer_cls):
    allow_unknown_subkeys = set(RLlibTrainer._allow_unknown_subkeys)
    trainer_cls(config=dict(num_workers=0))
    assert allow_unknown_subkeys == set(RLlibTrainer._allow_unknown_subkeys)


@pytest.fixture(
    params=({}, {"type": "two", "param": "default"}),
    ids="Unchanged OverrideType".split(),
)
def arbitrary(request):
    return request.param


def test_override_all_if_type_changes(trainer_cls, arbitrary):
    assert "arbitrary" in trainer_cls.options.allow_unknown_subkeys
    assert "arbitrary" in trainer_cls.options.override_all_if_type_changes
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
    assert not hasattr(trainer, "evaluation_metrics")

    res = trainer.train()
    assert "evaluation" in res
    assert not hasattr(trainer, "evaluation_metrics")

    # Assert evaluation is not run again
    old = res["evaluation"]
    new = trainer.train().get("evaluation", {})
    assert not new or set(old.keys()) == set(new.keys())
    assert not new or all(old[k] == new[k] for k in old.keys())
