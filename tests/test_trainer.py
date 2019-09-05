# pylint: disable=missing-docstring,redefined-outer-name
import ray


def setup_module():
    ray.init(object_store_memory=int(2e9))


def teardown_module():
    ray.shutdown()


def test_trainer_step(trainer_cls, env_creator):
    trainer = trainer_cls(env=env_creator)
    trainer.train()
