"""Mixins to add environment mimicking functions."""
import inspect
import warnings
from typing import Optional
from typing import Union

import raylab.envs as envs
from raylab.utils.types import DynamicsFn
from raylab.utils.types import RewardFn
from raylab.utils.types import TerminationFn


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

    def set_reward_from_config(self):
        """Build and set the reward function from environment configurations."""
        env_id, env_config = self.config["env"], self.config["env_config"]
        self.reward_fn = envs.get_reward_fn(env_id, env_config)
        self._set_reward_hook()

    def set_reward_from_callable(self, function: RewardFn):
        """Set the reward function from an external callable.

        Args:
            function: A callable that reproduces the environment's reward
                function
        """
        self._check_not_instance_method(function)
        self.reward_fn = function
        self._set_reward_hook()

    def set_termination_from_config(self):
        """Build and set a termination function from environment configurations."""
        env_id, env_config = self.config["env"], self.config["env_config"]
        self.termination_fn = envs.get_termination_fn(env_id, env_config)
        self._set_termination_hook()

    def set_termination_from_callable(self, function: TerminationFn):
        """Set the termination function from an external callable.

        Args:
            function: A callable that reproduces the environment's termination
                function
        """
        self._check_not_instance_method(function)
        self.termination_fn = function
        self._set_termination_hook()

    def set_dynamics_from_callable(self, function: DynamicsFn):
        """Set the dynamics function from an external callable.

        Args:
            function: A callable that reproduces the environment's dynamics
                function
        """
        self._check_not_instance_method(function)
        self.dynamics_fn = function
        self._set_dynamics_hook()

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

    @staticmethod
    def _check_not_instance_method(env_fn: Union[RewardFn, TerminationFn]):
        if inspect.ismethod(env_fn):
            warnings.warn(
                f"{env_fn.__name__} function is a bound instance method of"
                f" {env_fn.__self__}. Prefer static environment functions"
                " with markovian states that are not bounded to a particular"
                " instance"
            )
