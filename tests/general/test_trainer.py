# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import ray

from ..mock_env import MockEnv


def setup_module():
    ray.init()


def teardown_module():
    ray.shutdown()


@pytest.fixture(
    scope="module",
    params=(True, False),
    ids=(f"CompilePolicy({b})" for b in (True, False)),
)
def compile_policy(request):
    return request.param


@pytest.fixture(scope="module")
def trainer(trainer_cls, compile_policy):
    return trainer_cls(
        env=MockEnv,
        config={
            "compile_policy": compile_policy,
            "num_workers": 0,
            "rollout_fragment_length": 10,
            "evaluation_config": {"explore": False},
            "evaluation_interval": 1,
            "evaluation_num_episodes": 1,
            "evaluation_num_workers": 0,
        },
    )


@pytest.mark.slow
def test_trainer_step(trainer):
    trainer.train()


def test_trainer_eval(trainer):
    trainer._evaluate()


def test_trainer_restore(trainer):
    obj = trainer.save_to_object()
    trainer.restore_from_object(obj)
