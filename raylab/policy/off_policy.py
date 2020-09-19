# pylint:disable=missing-module-docstring
from abc import ABC
from abc import abstractmethod

from ray.rllib import SampleBatch

from raylab.options import option
from raylab.utils.annotations import TensorDict
from raylab.utils.replay_buffer import NumpyReplayBuffer

from .stats import learner_stats


def off_policy_options(cls: type) -> type:
    """Decorator to add default off-policy options used by OffPolicyMixin."""
    buffer_size = option(
        "buffer_size",
        default=int(1e4),
        help="""Size (number of transitions) of the replay buffer.""",
    )
    std_obs = option(
        "std_obs",
        default=False,
        help="Wheter to normalize replayed observations by the empirical mean and std.",
    )
    improvement_steps = option(
        "improvement_steps",
        default=1,
        help="""Policy improvement steps on each call to `learn_on_batch`.

        Example:
            With a 'rollout_fragment_length' of 1 and 'policy/improvement_steps' equal
            to 10,will perform 10 policy updates with minibatch size 'policy/batch_size'
            per environment step.
        """,
    )
    batch_size = option(
        "batch_size",
        default=128,
        help="Size of replay buffer batches sampled on each call to `improve_policy`.",
    )

    options = [buffer_size, std_obs, improvement_steps, batch_size]
    for opt in options:
        cls = opt(cls)

    return cls


class OffPolicyMixin(ABC):
    """Adds a replay buffer and standard procedures for `learn_on_batch`."""

    replay: NumpyReplayBuffer

    def build_replay_buffer(self):
        """Construct the experience replay buffer.

        Should be called by subclasses on init.
        """
        self.replay = NumpyReplayBuffer(
            self.observation_space, self.action_space, self.config["buffer_size"]
        )
        self.replay.seed(self.config["seed"])

    @learner_stats
    def learn_on_batch(self, samples: SampleBatch):
        """Run one logical iteration of training.

        Returns:
            An info dict from this iteration.
        """
        self.add_to_buffer(samples)

        info = {}
        info.update(self.get_exploration_info())

        for _ in range(int(self.config["improvement_steps"])):
            batch = self.replay.sample(self.config["batch_size"])
            batch = self.lazy_tensor_dict(batch)
            info.update(self.improve_policy(batch))

        return info

    def add_to_buffer(self, samples: SampleBatch):
        """Add sample batch to replay buffer"""
        self.replay.add(samples)
        if self.config["std_obs"]:
            self.replay.update_obs_stats()

    @abstractmethod
    def improve_policy(self, batch: TensorDict) -> dict:
        """Run one step of Policy Improvement."""
