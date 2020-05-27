"""Generic Trainer and base configuration for model-based agents."""
from ray.rllib.evaluation.metrics import get_learner_stats
from ray.rllib.utils import override

from raylab.agents.off_policy import GenericOffPolicyTrainer
from raylab.agents.off_policy import with_base_config as with_off_policy_config
from raylab.utils.dictionaries import deep_merge
from raylab.utils.replay_buffer import ReplayBuffer


BASE_CONFIG = with_off_policy_config(
    {
        # === Model Training ===
        # Number of minibatches to sample for dynamics model training loop
        "model_epochs": 120,
        # Size of minibatch for each dynamics model epoch
        "model_batch_size": 256,
        # === Policy Training ===
        # Number of policy improvement steps per real environment step
        "policy_improvements": 10,
        # Fraction of each policy minibatch to sample from environment replay pool
        "real_data_ratio": 0.1,
        # === Replay buffer ===
        # Size of the buffer for virtual samples
        "virtual_buffer_size": int(1e6),
        # number of model rollouts to add to augmented replay per real environment step
        "model_rollouts": 40,
    }
)


def with_base_config(config):
    """Returns the given config dict merged with the base model-based configuration."""
    return deep_merge(BASE_CONFIG, config, True)


class ModelBasedTrainer(GenericOffPolicyTrainer):
    """Generic trainer for model-based agents."""

    # pylint: disable=attribute-defined-outside-init

    @override(GenericOffPolicyTrainer)
    def _init(self, config, env_creator):
        super()._init(config, env_creator)

        self.virtual_replay = ReplayBuffer(
            config["virtual_buffer_size"], extra_keys=self._extra_replay_keys
        )

    @staticmethod
    @override(GenericOffPolicyTrainer)
    def validate_config(config):
        GenericOffPolicyTrainer.validate_config(config)
        assert (
            config["model_epochs"] >= 0
        ), "Cannot train model for a negative number of epochs"
        assert config["model_batch_size"] > 0, "Model batch size must be positive"
        assert (
            config["policy_improvements"] >= 0
        ), "Number of policy improvement steps must be non-negative"
        assert (
            0 <= config["real_data_ratio"] <= 1
        ), "Fraction of real data samples for policy improvement must be in [0, 1]"
        assert (
            config["virtual_buffer_size"] >= 0
        ), "Virtual buffer capacity must be non-negative"
        assert (
            config["model_rollouts"] >= 0
        ), "Cannot sample a negative number of model rollouts"

    @override(GenericOffPolicyTrainer)
    def _train(self):
        start_samples = self.sample_until_learning_starts()

        worker = self.workers.local_worker()
        while not self._iteration_done():
            samples = worker.sample()
            self.optimizer.num_steps_sampled += samples.count
            for row in samples.rows():
                self.replay.add(row)

            stats = self.train_dynamics_model()
            self.populate_virtual_buffer(samples.count)
            stats.update(self.improve_policy(samples.count))

        self.optimizer.num_steps_sampled += start_samples
        return self._log_metrics(stats)

    def train_dynamics_model(self):
        """Implements the model training loop.

        Calls the policy to optimize the model on each minibatch.
        """
        policy = self.workers.local_worker().get_policy()
        stats = {}

        for _ in range(self.config["model_epochs"]):
            batch = self.replay.sample(self.config["model_batch_size"])
            stats = get_learner_stats(policy.optimize_model(batch))

        return stats

    def populate_virtual_buffer(self, num_env_steps):
        """
        Add short model-generated rollouts branched from real data to the virtual pool.
        """
        if self.config["model_rollouts"] == 0:
            return
        policy = self.workers.local_worker().get_policy()

        real_samples = self.replay.sample(self.config["model_rollouts"] * num_env_steps)
        virtual_samples = policy.generate_virtual_sample_batch(real_samples)
        for row in virtual_samples.rows():
            self.virtual_replay.add(row)

    def improve_policy(self, num_env_steps):
        """Call the policy to perform policy improvement using the augmented replay."""
        policy = self.workers.local_worker().get_policy()
        batch_size = self.config["train_batch_size"]
        env_batch_size = int(batch_size * self.config["real_data_ratio"])
        model_batch_size = batch_size - env_batch_size

        stats = {}
        for _ in range(num_env_steps * self.config["policy_improvements"]):
            env_batch = self.replay.sample(env_batch_size)
            virtual_batch = self.virtual_replay.sample(model_batch_size)
            batch = env_batch.concat(virtual_batch)
            stats = get_learner_stats(policy.learn_on_batch(batch))
            self.optimizer.num_steps_trained += batch.count
        return stats
