"""Primitives for all Trainers."""
from ray.rllib.agents.trainer import Trainer as _Trainer


class Trainer(_Trainer):
    """Base Trainer for all algorithms. This should not be instantiated."""

    # pylint: disable=abstract-method,no-member,attribute-defined-outside-init,protected-access
    _allow_unknown_subkeys = _Trainer._allow_unknown_subkeys + ["module"]
