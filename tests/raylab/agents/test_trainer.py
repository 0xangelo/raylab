import pytest
from ray.rllib.agents.trainer import Trainer as RLlibTrainer

from raylab.agents.trainer import configure
from raylab.agents.trainer import option
from raylab.agents.trainer import Trainer


def test_dummy_policy(dummy_policy_cls, obs_space, action_space):
    policy = dummy_policy_cls(obs_space, action_space, {})
    assert policy.param == 0
    info = policy.learn_on_batch(None)
    assert "improved" in info
    assert info["improved"]
    assert policy.param == 1
    weights = policy.get_weights()
    assert "param" in weights
    assert weights["param"] == 1


@pytest.fixture(scope="module")
def trainer_cls(dummy_policy_cls):
    @configure
    @option("workers", False)
    @option("arbitrary/", allow_unknown_subkeys=True, override_all_if_type_changes=True)
    @option("arbitrary/type", "one")
    @option("arbitrary/key", "value")
    class MinimalTrainer(Trainer):
        _name = "MinimalTrainer"
        _policy = dummy_policy_cls

        def __init__(self, *args, **kwargs):
            super().__init__(*args, env="MockEnv", **kwargs)

        def _init(self, config, env_creator):
            if config["workers"]:
                self.workers = self._make_workers(
                    env_creator, self._policy, config, num_workers=config["num_workers"]
                )

        def step(self):
            info = {}
            if hasattr(self, "workers"):
                policy = self.get_policy()
                info.update(policy.learn_on_batch(None))

            return info

    return MinimalTrainer


@pytest.fixture(params=(True, False), ids=(f"Workers({b})" for b in (True, False)))
def workers(request):
    return request.param


def test_default_config(trainer_cls):
    options = trainer_cls.options

    assert "workers" in options.defaults
    assert "arbitrary" in options.defaults
    assert "arbitrary" in options.allow_unknown_subkeys
    assert "arbitrary" in options.override_all_if_type_changes


def test_wandb_config(trainer_cls):
    options = trainer_cls.options

    assert "wandb" in options.defaults
    assert isinstance(options.defaults["wandb"], dict)
    assert not options.defaults["wandb"]


def test_metrics_creation(trainer_cls, workers):
    should_have_workers = workers
    trainer = trainer_cls(config=dict(workers=workers, num_workers=0))

    assert not should_have_workers or hasattr(trainer, "metrics")


@pytest.fixture
def trainer(trainer_cls, workers):
    return trainer_cls(config=dict(workers=workers, num_workers=0))


def test_has_workers(trainer, workers):
    should_have_workers = workers
    assert not should_have_workers or hasattr(trainer, "workers")


def test_train(trainer, workers, trainable_info_keys):
    should_learn = workers
    if should_learn:
        info = trainer.train()
        assert trainer._iteration == 1
        assert trainer.iteration == 1

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


def test_returns_metrics(trainer_cls, workers):
    should_have_workers = workers
    trainer = trainer_cls(config=dict(workers=workers, num_workers=0))

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
