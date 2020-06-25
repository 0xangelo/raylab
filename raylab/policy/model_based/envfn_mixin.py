"""Mixins to add environment mimicking functions."""
from typing import Optional

from raylab.envs import get_reward_fn
from raylab.envs import get_termination_fn
from raylab.utils.annotations import DynamicsFn
from raylab.utils.annotations import RewardFn
from raylab.utils.annotations import TerminationFn


class EnvFnMixin:
    """Add methods to create and set environment functions.

    Allows user to choose methods for setting the reward, termination,
    and dynamics functions, which may vary by experiment setup.

    Attributes:
        reward_fn: callable mapping dynamics triplets to reward tensors
        termination_fn: callable mapping dynamics triplets to termination
            tensors
        dynamics_fn: callable mapping (state, action) pairs to (next-state,
            log-probability) pairs
    """

    reward_fn: Optional[RewardFn]
    termination_fn: Optional[TerminationFn]
    dynamics_fn: Optional[DynamicsFn]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_fn = None
        self.termination_fn = None
        self.dynamics_fn = None

    def _set_reward_hook(self):
        """Procedure to run upon setting the reward function.

        Subclasses should override this to do extra processing whenever
        `set_reward_from_callable` or `set_reward_from_config` is called.
        For example, the subclass may set up losses which assume access to
        environment functions.
        """

    def _set_termination_hook(self):
        """Procedure to run upon setting the termination function.

        Subclasses should override this to do extra processing whenever
        `set_termination_from_callable` or `set_termination_from_config` is
        called. For example, the subclass may set up losses which assume access
        to environment functions.
        """

    def _set_dynamics_hook(self):
        """Procedure to run upon setting the dynamics function.

        Subclasses should override this to do extra processing whenever
        `set_dynamics_from_callable` is called. For example, the subclass may
        set up losses which assume access to environment functions.
        """

    def set_reward_from_config(self, env_name: str, env_config: dict):
        """Build and set the reward function from environment configurations.

        Args:
            env_name: the environment's id
            env_config: the environment's configuration
        """
        self.reward_fn = get_reward_fn(env_name, env_config)
        self._set_reward_hook()

    def set_reward_from_callable(self, function: RewardFn):
        """Set the reward function from an external callable.

        Args:
            function: A callable that reproduces the environment's reward
                function
        """
        self.reward_fn = function
        self._set_reward_hook()

    def set_termination_from_config(self, env_name: str, env_config: dict):
        """Build and set the termination function from environment configurations.

        Args:
            env_name: the environment's id
            env_config: the environment's configuration
        """
        self.termination_fn = get_termination_fn(env_name, env_config)
        self._set_termination_hook()

    def set_termination_from_callable(self, function: TerminationFn):
        """Set the termination function from an external callable.

        Args:
            function: A callable that reproduces the environment's termination
                function
        """
        self.termination_fn = function
        self._set_termination_hook()

    def set_dynamics_from_callable(self, function: DynamicsFn):
        """Set the dynamics function from an external callable.

        Args:
            function: A callable that reproduces the environment's dynamics
                function
        """
        self.dynamics_fn = function
        self._set_dynamics_hook()
