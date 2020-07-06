"""Primitives for all Trainers."""
import copy
import textwrap
import warnings
from abc import ABCMeta
from typing import Callable
from typing import List
from typing import Optional

from ray.rllib.agents import with_common_config as with_rllib_config
from ray.rllib.agents.trainer import Trainer as _Trainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.optimizers import PolicyOptimizer
from ray.rllib.utils import override as overrd

from .config import Config
from .config import Info
from .config import Json
from .config import with_rllib_info


def config(
    key: str,
    default: Json,
    *,
    info: Optional[str] = None,
    override: bool = False,
    allow_unknown_subkeys: bool = False,
    override_all_if_type_changes: bool = False,
    separator: str = "/",
) -> Callable[[type], type]:
    """Returns a decorator for adding/overriding a Trainer class config.

    Args:
        key: Name of the config paremeter which the use can tune
        default: Default Jsonable value to set for the parameter
        info: Parameter help text explaining what the parameter does
        override: Whether to override an existing parameter
        allow_unknown_subkeys: Whether to allow new keys for dict parameters.
            This is only at the top level
        override_all_if_type_changes: Whether to override the entire value
            (dict) iff the 'type' key in this value dict changes. This is only
            at the top level
        separator: String token separating nested keys

    Raises:
        RuntimeError: If attempting to set an existing parameter with `override`
            set to `False`.
        RuntimeError: If attempting to override an existing parameter with its
            same default value.
        RuntimeError: If attempting to enable `allow_unknown_subkeys` or
            `override_all_if_type_changes` options for non-toplevel keys
    """
    # pylint:disable=protected-access,too-many-arguments
    if (allow_unknown_subkeys or override_all_if_type_changes) and separator in key:
        raise RuntimeError(
            "Cannot use 'allow_unknown_subkeys' or 'override_all_if_type_changes'"
            f" for non-toplevel key: '{key}'"
        )
    key_seq = key.split(separator)
    help_txt = info

    def add_config(cls):
        nonlocal key

        if allow_unknown_subkeys and not override:
            cls._allow_unknown_subkeys += [key]
        if override_all_if_type_changes and not override:
            cls._override_all_subkeys_if_type_changes += [key]

        config_, info_ = cls._default_config, cls._config_info
        for key in key_seq[:-1]:
            config_ = config_[key]
            if not isinstance(info_[key], dict):
                info_[key] = {"__help__": info_[key]}
            info_ = info_[key]
        key = key_seq[-1]

        if key in config_ and not override:
            raise RuntimeError(
                f"Attempted to override config key '{key}' but override=False."
            )
        if key in config_ and default == config_[key]:
            raise RuntimeError(
                f"Attempted to override config key {key} with the same value: {default}"
            )
        config_[key] = default

        if help_txt is not None:
            if key in info_ and not isinstance(info_[key], dict):
                info_[key] = {"__help__": info_[key]}
            info_[key] = textwrap.dedent(help_txt).rstrip()

        return cls

    return add_config


@config("compile_policy", False, info="Whether to optimize the policy's backend")
@config(
    "module", {}, info="Type and config of the PyTorch NN module.",
)
@config("torch_optimizer", {}, info="Config dict for PyTorch optimizers.")
class Trainer(_Trainer, metaclass=ABCMeta):
    """Base Trainer for all agents.

    Either a PolicyOptimizer or WorkerSet must be set (as `optimizer` or
    `workers` attributes respectively) to collect episode statistics.

    Always creates a PolicyOptimizer instance as the `optimizer` attribute to
    log episode metrics (to be removed in the future).

    Accessing `metrics` returns the optimizer so as to avoid confusion when
    updating metrics (e.g., `optimizer.num_steps_sampled`) even though a policy
    optimizer isn't being used by the algorithm.
    """

    evaluation_metrics: Optional[dict]
    optimizer: Optional[PolicyOptimizer]
    workers: Optional[WorkerSet]
    _allow_unknown_subkeys: List[str] = []
    _override_all_subkeys_if_type_changes: List[str] = []
    _config_info: Info = with_rllib_info({})
    _default_config: Config = with_rllib_config({})

    @overrd(_Trainer)
    def train(self):
        # Evaluate first, before any optimization is done
        if self._iteration == 0 and self.config["evaluation_interval"]:
            self.evaluation_metrics = self._evaluate()

        result = super().train()

        # Update global_vars after training so that they're saved if checkpointing
        self.global_vars["timestep"] = self.metrics.num_steps_sampled
        return result

    @classmethod
    def with_base_specs(cls, trainer_cls: type) -> type:
        """Decorator for using this class' config and info in the given trainer."""
        # pylint:disable=protected-access
        trainer_cls._default_config = copy.deepcopy(cls._default_config)
        trainer_cls._config_info = copy.deepcopy(cls._config_info)

        return trainer_cls

    def _setup(self, *args, **kwargs):
        for key in self._allow_unknown_subkeys:
            _Trainer._allow_unknown_subkeys += [key]
        for key in self._override_all_subkeys_if_type_changes:
            _Trainer._override_all_subkeys_if_type_changes += [key]
        try:
            super()._setup(*args, **kwargs)
        finally:
            for key in self._allow_unknown_subkeys:
                _Trainer._allow_unknown_subkeys.remove(key)
            for key in self._override_all_subkeys_if_type_changes:
                _Trainer._override_all_subkeys_if_type_changes.remove(key)

        # Always have a PolicyOptimizer to collect metrics
        if not hasattr(self, "optimizer"):
            if hasattr(self, "workers"):
                self.optimizer = PolicyOptimizer(self.workers)
            else:
                warnings.warn(
                    "No worker set initialized; episodes summary will be unavailable."
                )

        if self.config["compile_policy"]:
            if not hasattr(self, "workers"):
                raise RuntimeError(
                    f"{type(self).__name__} has no worker set. "
                    "Cannot access policies for compilation."
                )
            self.workers.foreach_policy(lambda p, _: p.compile())

    def __getattr__(self, attr):
        if attr == "metrics":
            return self.optimizer

        raise AttributeError(f"{type(self)} has not {attr} attribute")

    @overrd(_Trainer)
    def __getstate__(self):
        state = super().__getstate__()
        state["global_vars"] = self.global_vars
        return state

    @overrd(_Trainer)
    def __setstate__(self, state):
        self.global_vars = state["global_vars"]
        if hasattr(self, "workers"):
            self.workers.foreach_worker(lambda w: w.set_global_vars(self.global_vars))

        super().__setstate__(state)

    def _iteration_done(self, init_timesteps):
        return self.metrics.num_steps_sampled - init_timesteps >= max(
            self.config["timesteps_per_iteration"], 1
        )

    def _log_metrics(self, learner_stats, init_timesteps):
        res = self.collect_metrics()
        res.update(
            timesteps_this_iter=self.metrics.num_steps_sampled - init_timesteps,
            info=dict(learner=learner_stats, **res.get("info", {})),
        )
        if self._iteration == 0 and self.config["evaluation_interval"]:
            res.update(self.evaluation_metrics)
        return res
