"""Continuous Q-Learning with Normalized Advantage Functions."""
import time

from ray.rllib.utils.annotations import override
from ray.rllib.agents.trainer import Trainer, with_common_config
from ray.rllib.optimizers.replay_buffer import ReplayBuffer
from ray.rllib.policy.sample_batch import SampleBatch

from raylab.algorithms.naf.naf_policy import NAFTorchPolicy


DEFAULT_CONFIG = with_common_config(
    {
        "replay_buffer_size": int(1e5),
        "model": {"layers": [400, 300], "activation": "elu"},
        "torch_optimizer": "Adam",
        "torch_optimizer_options": {},
        "polyak": 0.995,
    }
)


class NAFTrainer(Trainer):
    """Single agent trainer for NAF."""

    _name = "NAF"
    _default_config = DEFAULT_CONFIG
    _policy = NAFTorchPolicy

    @override(Trainer)
    def _init(self, config, env_creator):
        # pylint: disable=attribute-defined-outside-init
        self._validate_config(config)
        policy_cls = self._policy
        self.workers = self._make_workers(
            env_creator, policy_cls, config, num_workers=0
        )
        self.replay = ReplayBuffer(config["replay_buffer_size"])

    @override(Trainer)
    def _train(self):
        worker = self.workers.local_worker()
        policy = worker.get_policy()

        start = time.time()
        num_steps_sampled = 0
        while True:
            samples = worker.sample()
            num_steps_sampled += samples.count
            for row in samples.rows():
                self.replay.add(
                    row[SampleBatch.CUR_OBS],
                    row[SampleBatch.ACTIONS],
                    row[SampleBatch.REWARDS],
                    row[SampleBatch.NEXT_OBS],
                    row[SampleBatch.DONES],
                    weight=None,
                )

            for _ in range(samples.count):
                batch = self.replay.sample(self.config["train_batch_size"])
                fetches = policy.learn_on_batch(batch)

            if (
                time.time() - start >= self.config["min_iter_time_s"]
                and num_steps_sampled >= self.config["timesteps_per_iteration"]
            ):
                break

        return fetches

    @staticmethod
    def _validate_config(config):
        assert config["num_workers"] >= 0, "No point in using additional workers."
