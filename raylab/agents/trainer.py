"""Primitives for all Trainers."""
import copy
import warnings
from abc import ABCMeta
from typing import Callable
from typing import Optional

from ray.rllib.agents import with_common_config as with_rllib_config
from ray.rllib.agents.trainer import Trainer as _Trainer
from ray.rllib.agents.trainer import with_base_config
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.optimizers import PolicyOptimizer
from ray.rllib.utils import override as overrd

from .config import Config
from .config import Info
from .config import Json
from .config import with_rllib_info

_Trainer._allow_unknown_subkeys += ["module", "torch_optimizer"]
_Trainer._override_all_subkeys_if_type_changes += ["module"]


BASE_CONFIG = with_rllib_config(
    {"compile_policy": False, "module": {}, "torch_optimizer": {}}
)

BASE_INFO = with_rllib_info(
    {
        "compile_policy": "Whether to optimize the policy's backend",
        "module": "Type and config of the PyTorch NN module.",
        "torch_optimizer": "Config dict for PyTorch optimizers.",
    }
)


def with_common_config(extra_config: Config) -> Config:
    """Returns the given config dict merged with common agent confs."""
    return with_base_config(BASE_CONFIG, extra_config)


def config(
    key: str,
    default: Json,
    *,
    info: Optional[str] = None,
    override: bool = False,
    separator: str = "/",
) -> Callable[[type], type]:
    """Decorator for adding/overriding a config to a Trainer class.

    Args:
        key: Name of the config paremeter which the use can tune
        default: Default Jsonable value to set for the parameter
        info: Parameter help text explaining what the parameter does
        override: Whether to override an existing parameter
        separator: String token separating nested keys

    Raises:
        RuntimeError: If attempting to set an existing parameter with `override`
            set to `False`.
        RuntimeError: If attempting to override an existing parameter with its
            same default value.
    """
    # pylint:disable=protected-access
    key_seq = key.split(separator)

    def add_config(cls):
        nonlocal info

        conf_, info_ = cls._default_config, cls._config_info
        for key in key_seq[:-1]:
            conf_ = conf_[key]
            if not isinstance(info_[key], dict):
                info_[key] = {"__help__": info_[key]}
            info_ = info_[key]

        key = key_seq[-1]
        if key in conf_ and not override:
            raise RuntimeError(
                f"Attempted to override config key '{key}' but override=False."
            )
        if key in conf_ and default == conf_[key]:
            raise RuntimeError(
                f"Attempted to override config key {key} with the same value: {default}"
            )

        conf_[key] = default
        if info is not None:
            info_[key] = info

        return cls

    return add_config


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
    _config_info: Info = BASE_INFO
    _default_config: Config = BASE_CONFIG

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
        super()._setup(*args, **kwargs)
        # Have a PolicyOptimizer by default to collect metrics
        if not hasattr(self, "optimizer"):
            if hasattr(self, "workers"):
                self.optimizer = PolicyOptimizer(self.workers)
            else:
                warnings.warn(
                    "No worker set initialized; episodes summary will be unavailable."
                )

        if self.config["compile_policy"]:
            if hasattr(self, "workers"):
                workers = self.workers
            else:
                raise RuntimeError(
                    f"{type(self).__name__} has no worker set. "
                    "Cannot access policies for compilation."
                )
            workers.foreach_policy(lambda p, _: p.compile())

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
