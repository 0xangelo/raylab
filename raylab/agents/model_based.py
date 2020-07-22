"""Generic Trainer and base configuration for model-based agents."""
from typing import List
from typing import Tuple

from ray.rllib import SampleBatch
from ray.rllib.utils import override

import raylab.envs as envs
from raylab.agents import trainer
from raylab.agents.off_policy import OffPolicyTrainer
from raylab.utils.annotations import StatDict
from raylab.utils.replay_buffer import NumpyReplayBuffer


@trainer.configure
@trainer.option(
    "holdout_ratio",
    0.2,
    help="Fraction of replay buffer to use as validation dataset"
    " (hence not for training)",
)
@trainer.option(
    "max_holdout", 5000, help="Maximum number of samples to use as validation dataset"
)
class ModelBasedTrainer(OffPolicyTrainer):
    """Generic trainer for model-based agents.

    Sets reward and termination functions for policies. These functions must be
    either:
    * Registered via `raylab.envs.register_reward_fn` and
      `raylab.envs.register_termination_fn`
    * Accessible attributes of the environment as `reward_fn` and
      `termination_fn`

    Policies must implement `optimize_model` according to
    `raylab.policy:ModelTrainingMixin`
    """

    @override(OffPolicyTrainer)
    def _init(self, config, env_creator):
        super()._init(config, env_creator)
        policy = self.get_policy()
        worker = self.workers.local_worker()

        if envs.has_reward_fn(config["env"]):
            policy.set_reward_from_config(config["env"], config["env_config"])
        else:
            policy.set_reward_from_callable(worker.env.reward_fn)

        if envs.has_termination_fn(config["env"]):
            policy.set_termination_from_config(config["env"], config["env_config"])
        else:
            policy.set_termination_from_callable(worker.env.termination_fn)

    @staticmethod
    @override(OffPolicyTrainer)
    def validate_config(config):
        OffPolicyTrainer.validate_config(config)
        assert (
            config["holdout_ratio"] < 1.0
        ), "Holdout data cannot be the entire dataset"
        assert (
            config["max_holdout"] >= 0
        ), "Maximum number of holdout samples must be non-negative"
        assert (
            config["policy_improvements"] >= 0
        ), "Number of policy improvement steps must be non-negative"

    @override(OffPolicyTrainer)
    def _train(self):
        pre_learning_steps = self.sample_until_learning_starts()
        timesteps_this_iter = 0

        config = self.config
        worker = self.workers.local_worker()
        stats = {}
        while timesteps_this_iter < max(self.config["timesteps_per_iteration"], 1):
            samples = worker.sample()
            timesteps_this_iter += samples.count
            for row in samples.rows():
                self.replay.add(row)

            _, model_train_info = self.train_dynamics_model()
            policy_train_info = self.improve_policy(
                config["policy_improvements"] * samples.count
            )

            stats.update(model_train_info)
            stats.update(policy_train_info)

        self.metrics.num_steps_sampled += timesteps_this_iter
        return self._log_metrics(stats, timesteps_this_iter + pre_learning_steps)

    def train_dynamics_model(self) -> Tuple[List[float], StatDict]:
        """Implements the model training step.

        Calls the policy to optimize the model on the environment replay buffer.

        Returns:
            A tuple containing the list of evaluation losses for each model and
            a dictionary of training statistics
        """
        samples = self.replay.all_samples()
        samples.shuffle()
        holdout = min(
            int(len(self.replay) * self.config["holdout_ratio"]),
            self.config["max_holdout"],
        )
        train_data, eval_data = samples.slice(holdout, None), samples.slice(0, holdout)
        eval_data = None if eval_data.count == 0 else eval_data

        policy = self.get_policy()
        eval_losses, stats = policy.optimize_model(train_data, eval_data)

        return eval_losses, stats

    def improve_policy(self, num_improvements: int) -> StatDict:
        """Improve the policy on previously collected environment data.

        Calls the policy to learn on batches samples from the replay buffer.

        Args:
            num_improvements: number of times to call `policy.learn_on_batch`

        Returns:
            A dictionary of training and exploration statistics
        """
        policy = self.get_policy()
        batch_size = self.config["train_batch_size"]

        stats = {}
        for _ in range(num_improvements):
            batch = self.replay.sample(batch_size)
            stats.update(policy.learn_on_batch(batch))
            self.metrics.num_steps_trained += batch.count

        stats.update(policy.get_exploration_info())
        return stats


@trainer.configure
@trainer.option(
    "virtual_buffer_size", int(1e6), help="Size of the buffer for virtual samples"
)
@trainer.option(
    "model_rollouts",
    40,
    help="Populate virtual replay with this many model rollouts per environment step",
)
@trainer.option(
    "real_data_ratio",
    0.1,
    help="Fraction of each policy minibatch to sample from environment replay pool",
)
class DynaLikeTrainer(ModelBasedTrainer):
    """Generic trainer for model-based agents with dyna-like data augmentation.

    If `model_rollouts` > 0, policies must implement `setup_sampling_models`
    and `generate_virtual_sample_batch` according to
    `raylab.policy:ModelSamplingMixin`
    """

    # pylint:disable=attribute-defined-outside-init

    @staticmethod
    def validate_config(config):
        ModelBasedTrainer.validate_config(config)
        assert (
            0 <= config["real_data_ratio"] <= 1
        ), "Fraction of real data samples for policy improvement must be in [0, 1]"
        assert (
            config["virtual_buffer_size"] >= 0
        ), "Virtual buffer capacity must be non-negative"
        assert (
            config["model_rollouts"] >= 0
        ), "Cannot sample a negative number of model rollouts"

    @override(OffPolicyTrainer)
    def build_replay_buffer(self, config):
        super().build_replay_buffer(config)
        policy = self.get_policy()
        self.virtual_replay = NumpyReplayBuffer(
            policy.observation_space, policy.action_space, config["virtual_buffer_size"]
        )
        self.virtual_replay.seed(config["seed"])

    @override(ModelBasedTrainer)
    def _train(self):
        pre_learning_steps = self.sample_until_learning_starts()
        timesteps_this_iter = 0

        config = self.config
        worker = self.workers.local_worker()
        stats = {}
        while timesteps_this_iter < max(self.config["timesteps_per_iteration"], 1):
            samples = worker.sample()
            timesteps_this_iter += samples.count
            for row in samples.rows():
                self.replay.add(row)

            eval_losses, model_train_info = self.train_dynamics_model()
            self.populate_virtual_buffer(
                eval_losses, config["model_rollouts"] * samples.count
            )
            policy_train_info = self.improve_policy(
                config["policy_improvements"] * samples.count
            )

            stats.update(model_train_info)
            stats.update(policy_train_info)

        self.metrics.num_steps_sampled += timesteps_this_iter
        return self._log_metrics(stats, timesteps_this_iter + pre_learning_steps)

    def populate_virtual_buffer(self, eval_losses: List[float], num_rollouts: int):
        """Add model rollouts branched from real data to the virtual pool.

        Args:
            eval_losses: the latest validation losses for each model in the
                ensemble
            num_rollouts: number of initial states to samples from the
                environment replay buffer
        """
        if not (num_rollouts and self.config["real_data_ratio"] < 1.0):
            return

        policy = self.get_policy()
        policy.setup_sampling_models(eval_losses)

        real_samples = self.replay.sample(num_rollouts)
        virtual_samples = policy.generate_virtual_sample_batch(real_samples)
        for row in virtual_samples.rows():
            self.virtual_replay.add(row)

    def improve_policy(self, num_improvements: int) -> StatDict:
        """Improve the policy on a mixture of environment and model data.

        Calls the policy to learn on batches sampled from the environment and
        model rollouts.

        Args:
            num_improvements: number of times to call `policy.learn_on_batch`

        Returns:
            A dictionary of training and exploration statistics
        """
        policy = self.get_policy()
        batch_size = self.config["train_batch_size"]
        env_batch_size = int(batch_size * self.config["real_data_ratio"])
        model_batch_size = batch_size - env_batch_size

        stats = {}
        for _ in range(num_improvements):
            samples = []
            if env_batch_size:
                samples += [self.replay.sample(env_batch_size)]
            if model_batch_size:
                samples += [self.virtual_replay.sample(model_batch_size)]
            batch = SampleBatch.concat_samples(samples)
            stats.update(policy.learn_on_batch(batch))
            self.metrics.num_steps_trained += batch.count

        stats.update(policy.get_exploration_info())
        return stats
