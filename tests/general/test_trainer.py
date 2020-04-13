# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import ray

from ..mock_env import MockEnv


def setup_module():
    ray.init()


def teardown_module():
    ray.shutdown()


@pytest.mark.slow
def test_trainer_step(trainer_cls):
    trainer = trainer_cls(env=MockEnv, config={"num_workers": 0})
    trainer.train()
