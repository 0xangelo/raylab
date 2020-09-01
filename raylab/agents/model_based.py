"""Generic Trainer and base configuration for model-based agents."""
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from ray.rllib import Policy
from ray.rllib import SampleBatch
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils import override

from raylab.agents import trainer
from raylab.agents.off_policy import OffPolicyTrainer
from raylab.utils.annotations import StatDict
from raylab.utils.replay_buffer import NumpyReplayBuffer
from raylab.utils.timer import TimerStat


logger = logging.getLogger(__name__)


def set_policy_with_env_fn(worker_set: WorkerSet, fn_type: str = "reward"):
    """Set the desired environment function for all policies in the worker set.

    Args:
        worker_set: A worker set instance, usually from a trainer
        fn_type: The type of environment function, either 'reward',
            'termination', or 'dynamics'
        from_env: Whether to retrieve the function from the environment instance
            or from the global registry
    """
    worker_set.foreach_worker(
        lambda w: w.foreach_policy(
            lambda p, _: _set_from_env_if_possible(p, w.env, fn_type)
        )
    )


def _set_from_env_if_possible(policy: Policy, env: Any, fn_type: str = "reward"):
    env_fn = getattr(env, fn_type + "_fn", None)
    if fn_type == "reward":
        if env_fn:
            policy.set_reward_from_callable(env_fn)
        else:
            policy.set_reward_from_config()
    elif fn_type == "termination":
        if env_fn:
            policy.set_termination_from_callable(env_fn)
        else:
            policy.set_termination_from_config()
    elif fn_type == "dynamics":
        if env_fn:
            policy.set_dynamics_from_callable(env_fn)
        else:
            raise ValueError(
                f"Environment '{env}' has no '{fn_type + '_fn'}' attribute"
            )
    else:
        raise ValueError(f"Invalid env function type '{fn_type}'")


@trainer.configure
@trainer.option(
    "model_update_interval",
    default=1,
    help="""Number of calls to rollout worker between each model update run.

    Will collect this many rollout fragments of length 'rollout_fragment_length'
    between calls to `train_dynamics_model`

    Example:
        With a 'rollout_fragment_length' of 1 and 'model_update_interval' of 25,
        will collect 25 environment transitions between each model optimization
        loop.
    """,
)
@trainer.option(
    "policy_improvement_interval",
    default=1,
    help="""Number of rollout worker calls between each `policy.learn_on_batch` call.

    Uses the same semantics as 'model_update_interval'.
    """,
)
class ModelBasedTrainer(OffPolicyTrainer):
    """Generic trainer for model-based agents.

    Sets reward and termination functions for policies. These functions must be
    either:
    * Registered via `raylab.envs.register_reward_fn` and
      `raylab.envs.register_termination_fn`
    * Accessible attributes of the environment as `reward_fn` and
      `termination_fn`. These should not be bound instance methods; all
      necessary information should be encoded in the inputs, (state, action,
      and next state) i.e., the states should be markovian.

    Policies must implement `optimize_model` according to
    `raylab.policy:ModelTrainingMixin`
    """

    # pylint:disable=attribute-defined-outside-init
    timers: Dict[str, TimerStat]

    @override(OffPolicyTrainer)
    def _init(self, config, env_creator):
        super()._init(config, env_creator)
        set_policy_with_env_fn(self.workers, fn_type="reward")
        set_policy_with_env_fn(self.workers, fn_type="termination")

        self.timers = {"model": TimerStat(), "policy": TimerStat()}
        self._sample_calls: int = 0

    @staticmethod
    @override(OffPolicyTrainer)
    def validate_config(config):
        OffPolicyTrainer.validate_config(config)

        msg = "'{key}' must be a positive number of RolloutWorker sample calls"
        for key in "model_update_interval policy_improvement_interval".split():
            assert config[key] > 0, msg.format(key=key)

    @override(OffPolicyTrainer)
    def sample_until_learning_starts(self) -> int:
        sample_count = super().sample_until_learning_starts()
        if sample_count:
            logger.info("Starting model warmup")
            _, warmup_stats = self.train_dynamics_model(warmup=True)
            # pylint:disable=logging-too-many-args
            logger.info("Finished model warmup with stats: %s", warmup_stats)

        return sample_count

    @override(OffPolicyTrainer)
    def single_iteration(self) -> Tuple[int, dict]:
        info = {}

        samples = self.worker.sample()
        self.update_steps_sampled(samples.count)
        self.add_to_buffer(samples)
        self._sample_calls += 1

        if self._sample_calls % self.config["model_update_interval"] == 0:
            with self.timers["model"] as timer:
                _, model_info = self.train_dynamics_model(warmup=False)
                timer.push_units_processed(model_info["model_epochs"])

            info.update(model_info)

        if self._sample_calls % self.config["policy_improvement_interval"] == 0:
            with self.timers["policy"] as timer:
                times = self.config["policy_improvements"]
                policy_info = self.improve_policy(times=times)
                timer.push_units_processed(times)

            info.update(policy_info)

        return samples.count, info

    def train_dynamics_model(
        self, warmup: bool = False
    ) -> Tuple[List[float], StatDict]:
        """Implements the model training step.

        Calls the policy to optimize the model on the environment replay buffer.

        Args:
            warmup: Whether the optimization is being done on data collected
                via :meth:`sample_until_learning_starts`.

        Returns:
            A tuple containing the list of evaluation losses for each model and
            a dictionary of training statistics
        """
        samples = self.replay.all_samples()
        eval_losses, stats = self.get_policy().optimize_model(samples, warmup=warmup)
        return eval_losses, stats

    def improve_policy(self, times: int) -> StatDict:
        """Improve the policy on previously collected environment data.

        Calls the policy to learn on batches samples from the replay buffer.

        Args:
            times: number of times to call `policy.learn_on_batch`

        Returns:
            A dictionary of training and exploration statistics
        """
        policy = self.get_policy()
        batch_size = self.config["train_batch_size"]

        stats = {}
        for _ in range(times):
            batch = self.replay.sample(batch_size)
            stats.update(policy.learn_on_batch(batch))
            self.metrics.num_steps_trained += batch.count

        stats.update(policy.get_exploration_info())
        return stats

    @override(OffPolicyTrainer)
    def _log_metrics(self, learner_stats, timesteps_this_iter):
        metrics = super()._log_metrics(learner_stats, timesteps_this_iter)
        model_timer = self.timers["model"]
        policy_timer = self.timers["policy"]
        metrics.update(
            model_time_s=round(model_timer.mean, 3),
            policy_time_s=round(policy_timer.mean, 3),
            # Get mean number of model epochs per second spent updating the model
            model_update_throughput=round(model_timer.mean_throughput, 3),
            # Get mean number of policy updates per second spent updating the policy
            policy_update_throughput=round(policy_timer.mean_throughput, 3),
        )
        return metrics


@trainer.configure
@trainer.option(
    "virtual_buffer_size",
    default=int(1e6),
    help="Size of the buffer for virtual samples",
)
@trainer.option(
    "model_rollouts",
    default=40,
    help="""Number of model rollouts to add to virtual buffer each policy interval.

    Populates virtual replay with this many model rollouts before each policy
    improvement.
    """,
)
@trainer.option(
    "real_data_ratio",
    default=0.1,
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
    def single_iteration(self):
        info = {}

        samples = self.worker.sample()
        self.update_steps_sampled(samples.count)
        self.add_to_buffer(samples)
        self._sample_calls += 1

        if self._sample_calls % self.config["model_update_interval"] == 0:
            with self.timers["model"] as timer:
                eval_losses, model_info = self.train_dynamics_model(warmup=False)
                timer.push_units_processed(model_info["model_epochs"])

            self.get_policy().set_new_elite(eval_losses)
            info.update(model_info)

        if self._sample_calls % self.config["policy_improvement_interval"] == 0:
            self.populate_virtual_buffer(self.config["model_rollouts"])

            with self.timers["policy"] as timer:
                times = self.config["policy_improvements"]
                policy_info = self.improve_policy(times=times)
                timer.push_units_processed(times)

            info.update(policy_info)

        return samples.count, info

    def populate_virtual_buffer(self, num_rollouts: int):
        """Add model rollouts branched from real data to the virtual pool.

        Args:
            num_rollouts: number of initial states to samples from the
                environment replay buffer
        """
        if not (num_rollouts and self.config["real_data_ratio"] < 1.0):
            return

        real_samples = self.replay.sample(num_rollouts)
        policy = self.get_policy()
        virtual_samples = policy.generate_virtual_sample_batch(real_samples)
        for row in virtual_samples.rows():
            self.virtual_replay.add(row)

    def improve_policy(self, times: int) -> StatDict:
        """Improve the policy on a mixture of environment and model data.

        Calls the policy to learn on batches sampled from the environment and
        model rollouts.

        Args:
            times: number of times to call `policy.learn_on_batch`

        Returns:
            A dictionary of training and exploration statistics
        """
        policy = self.get_policy()
        batch_size = self.config["train_batch_size"]
        env_batch_size = int(batch_size * self.config["real_data_ratio"])
        model_batch_size = batch_size - env_batch_size

        stats = {}
        for _ in range(times):
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
