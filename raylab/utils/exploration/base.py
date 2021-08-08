"""Base implementations for all exploration strategies."""
import textwrap
from abc import ABCMeta, abstractmethod
from typing import Optional

from ray.rllib.utils.exploration import Exploration
from torch import nn

Model = Optional[nn.Module]


class IncompatibleExplorationError(Exception):
    """Exception raised for incompatible exploration and NN module.

    Args:
        exp_cls: Exploration class
        module: NN module
        err: AssertionError explaining the reason why exploration and module are
            incompatible

    Attributes:
        message: Human-readable text explaining what caused the incompatibility
    """

    def __init__(self, exp_cls: type, module: Model, err: Exception):
        # pylint:disable=unused-argument
        msg = f"""\
        Exploration type {exp_cls} is incompatible with NN module of type
        {type(module)}. Reason:
            {err}
        """
        super().__init__(textwrap.dedent(msg))
        self.message = msg


class BaseExploration(Exploration, metaclass=ABCMeta):
    """Base class for exploration objects."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.check_model_compat(self.model.module)
        except AssertionError as err:
            raise IncompatibleExplorationError(
                type(self), self.model.module, err
            ) from err

    @classmethod
    @abstractmethod
    def check_model_compat(cls, model: Model):
        """Assert the given NN module is compatible with the exploration.

        Raises:
            IncompatibleDistClsError: If `model` is incompatible with the
                exploration class
        """
