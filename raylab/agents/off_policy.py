# pylint: disable=missing-docstring
# pylint: enable=missing-docstring
from ray.rllib.utils.annotations import override
from ray.rllib.evaluation.metrics import get_learner_stats
from ray.rllib.optimizers import PolicyOptimizer

from raylab.agents import Trainer
from raylab.utils.replay_buffer import ReplayBuffer


class GenericOffPolicyTrainer(Trainer):
    """Generic trainer for off-policy agents."""

    # pylint: disable=attribute-defined-outside-init

    _name = ""
    _default_config = None
    _policy = None

    @override(Trainer)
    def _init(self, config, env_creator):
        self._validate_config(config)
        self.workers = self._make_workers(
            env_creator, self._policy, config, num_workers=0
        )
        # Dummy optimizer to log stats
        self.optimizer = PolicyOptimizer(self.workers)
        self.replay = ReplayBuffer(config["buffer_size"])

    @override(Trainer)
    def _train(self):
        worker = self.workers.local_worker()
        policy = worker.get_policy()

        while not self._iteration_done():
            samples = worker.sample()
            self.optimizer.num_steps_sampled += samples.count
            for row in samples.rows():
                self.replay.add(row)

            for _ in range(samples.count):
                batch = self.replay.sample(self.config["train_batch_size"])
                stats = get_learner_stats(policy.learn_on_batch(batch))
                self.optimizer.num_steps_trained += batch.count

        return self._log_metrics(stats)

    @staticmethod
    def _validate_config(config):
        assert config["num_workers"] == 0, "No point in using additional workers."
        assert (
            config["rollout_fragment_length"] >= 1
        ), "At least one sample must be collected."
        assert (
            config["batch_mode"] == "complete_episodes"
        ), "Must sample complete episodes."
