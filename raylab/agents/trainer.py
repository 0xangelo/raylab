"""Primitives for all Trainers."""
import copy
import inspect
import warnings
from abc import ABCMeta
from collections import namedtuple
from functools import partial
from typing import Callable
from typing import List
from typing import Optional

try:
    import wandb
except ImportError:
    wandb = None

from ray.rllib.agents import with_common_config as with_rllib_config
from ray.rllib.agents.trainer import Trainer as RLlibTrainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.optimizers import PolicyOptimizer
from ray.rllib.utils import override as overrd

from .config import Config
from .config import Info
from .config import Json
from .config import with_rllib_info


# ==============================================================================
# Programatic config setting
# ==============================================================================
ConfSetter = namedtuple("ConfSetter", "key setter")


def configure(cls: type) -> type:
    """Decorator for finishing the configuration setup for a Trainer class.

    Should be called after all :func:`config` decorators have been applied,
    i.e., as the top-most decorator.
    """
    # pylint:disable=protected-access
    cls._default_config = copy.deepcopy(cls._default_config)
    cls._config_info = copy.deepcopy(cls._config_info)
    cls._allow_unknown_subkeys = copy.deepcopy(cls._allow_unknown_subkeys)
    cls._override_all_subkeys_if_type_changes = copy.deepcopy(
        cls._override_all_subkeys_if_type_changes
    )

    for conf_setter in sorted(cls._to_set, key=lambda x: x.key):
        conf_setter.setter(cls)
    cls._to_set = None

    return cls


def option(
    key: str,
    default: Json,
    *,
    help: Optional[str] = None,
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
    # pylint:disable=too-many-arguments,redefined-builtin
    if (allow_unknown_subkeys or override_all_if_type_changes) and separator in key:
        raise RuntimeError(
            "Cannot use 'allow_unknown_subkeys' or 'override_all_if_type_changes'"
            f" for non-toplevel key: '{key}'"
        )

    setter = ConfSetter(
        key=key,
        setter=partial(
            _set_config,
            key=key,
            default=default,
            info=help,
            override=override,
            allow_unknown_subkeys=allow_unknown_subkeys,
            override_all_if_type_changes=override_all_if_type_changes,
            separator=separator,
        ),
    )

    def add_setter(cls):
        # pylint:disable=protected-access
        if cls._to_set is None:
            cls._to_set = []
        cls._to_set += [setter]
        return cls

    return add_setter


def _set_config(
    cls: type,
    key: str,
    default: Json,
    info: Optional[str] = None,
    override: bool = False,
    allow_unknown_subkeys: bool = False,
    override_all_if_type_changes: bool = False,
    separator: str = "/",
):
    # pylint:disable=too-many-arguments,protected-access
    key_seq = key.split(separator)

    if allow_unknown_subkeys and not override:
        cls._allow_unknown_subkeys += [key]
    if override_all_if_type_changes and not override:
        cls._override_all_subkeys_if_type_changes += [key]

    config_, info_ = cls._default_config, cls._config_info
    for key_ in key_seq[:-1]:
        config_ = config_.setdefault(key_, {})
        info_ = info_.setdefault(key_, {})
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

    if info is not None:
        help_txt = inspect.cleandoc(info)
        if isinstance(config_[key], dict):
            info_[key] = {"__help__": help_txt}
        else:
            info_[key] = help_txt


# ==============================================================================
# Base Raylab Trainer
# ==============================================================================
Jsonable = (dict, list, str, int, float, bool, type(None))


@configure
@option(
    "wandb",
    {},
    help="""Configs for integration with Weights & Biases.

    Don't forget to:
      * install `wandb` via pip
      * login to W&B with the appropriate API key for your
        team/project.

    Check out the Quickstart for more information:
    `https://docs.wandb.com/quickstart`
    """,
)
@option(
    "wandb/project", None, help="The name of the project to which this run will belong"
)
@option(
    "wandb/entity",
    None,
    help="""The team posting this run (default: your username or your default team).

    This should be set if the project does not belong to the default team set
    via `wandb init` in the command line.
    """,
)
@option(
    "wandb/config_exclude_keys",
    (),
    help="""String keys to exclude storing in W&B when specifying config.

    Only works for toplevel config keys. Can be used to avoid raising errors
    when trying to log unJsonable hyperparameters to W&B.

    Always ignores `wandb` and `callbacks` configs.
    """,
)
@option("compile_policy", False, help="Whether to optimize the policy's backend")
@option(
    "module",
    {},
    help="Type and config of the PyTorch NN module.",
    allow_unknown_subkeys=True,
    override_all_if_type_changes=True,
)
@option(
    "torch_optimizer",
    {},
    help="Config dict for PyTorch optimizers.",
    allow_unknown_subkeys=True,
)
class Trainer(RLlibTrainer, metaclass=ABCMeta):
    """Base Trainer for all agents.

    Either a PolicyOptimizer or WorkerSet must be set (as `optimizer` or
    `workers` attributes respectively) to collect episode statistics.

    Always creates a PolicyOptimizer instance as the `optimizer` attribute to
    log episode metrics (to be removed in the future).

    Accessing `metrics` returns the optimizer, which is useful when using an
    optimizer only to log metrics. This way, the user can update
    `self.metrics.num_steps_sampled` and the results will be logged via RLlib's
    framework.

    Integration with `Weights & Biases`_ is available. The user must install
    `wandb` and set a project name in the `wandb` subconfig dict.

    .. _`Weights & Biases`: https://docs.wandb.com/
    """

    optimizer: Optional[PolicyOptimizer]
    workers: Optional[WorkerSet]

    _to_set: Optional[List[ConfSetter]] = None
    _config_info: Info = with_rllib_info({})
    _default_config: Config = with_rllib_config({})

    @overrd(RLlibTrainer)
    def train(self):
        result = {}

        # Run evaluation once before any optimization is done
        if self._iteration == 0 and self.config["evaluation_interval"]:
            result.update(self._evaluate())

        result.update(super().train())

        # Update global_vars after training so that they're saved if checkpointing
        self.global_vars["timestep"] = self.metrics.num_steps_sampled

        if self.config["wandb"]["project"]:
            self._wandb_log_result(result)
        return result

    @staticmethod
    def _wandb_log_result(result: dict):
        # Avoid logging the config every iteration
        # Only log Jsonable objects
        filtered = {
            k: v for k, v in result.items() if k != "config" and isinstance(v, Jsonable)
        }
        wandb.log(filtered)

    def _setup(self, *args, **kwargs):
        if self._to_set:
            raise RuntimeError(
                f"{type(self).__name__} still has configs to be set."
                " Did you call `trainer.configure` as the last decorator?"
            )

        cls_attrs = ("_allow_unknown_subkeys", "_override_all_subkeys_if_type_changes")
        attr_cache = ((attr, getattr(RLlibTrainer, attr)) for attr in cls_attrs)
        for attr in cls_attrs:
            setattr(RLlibTrainer, attr, getattr(self, attr))
        try:
            super()._setup(*args, **kwargs)
        finally:
            for attr, cache in attr_cache:
                setattr(RLlibTrainer, attr, cache)

        # Always have a PolicyOptimizer to collect metrics
        if not hasattr(self, "optimizer"):
            if hasattr(self, "workers"):
                self.optimizer = PolicyOptimizer(self.workers)
            else:
                warnings.warn(
                    "No worker set initialized; episodes summary will be unavailable."
                )

        # Always have a WorkerSet to get workers and policy
        if hasattr(self, "optimizer") and not hasattr(self, "workers"):
            self.workers = self.optimizer.workers

        if self.config["compile_policy"]:
            if not hasattr(self, "workers"):
                raise RuntimeError(
                    f"{type(self).__name__} has no worker set. "
                    "Cannot access policies for compilation."
                )
            self.workers.foreach_policy(lambda p, _: p.compile())

        if self.config["wandb"]["project"]:
            self._setup_wandb()

    def _setup_wandb(self):
        assert wandb is not None, "Unable to import wandb, did you install it via pip?"

        config_exclude_keys = {"wandb", "callbacks"}
        config_exclude_keys.update(self.config["wandb"]["config_exclude_keys"])
        wandb.init(
            name=self._name,
            project=self.config["wandb"]["project"],
            entity=self.config["wandb"]["entity"],
            config_exclude_keys=config_exclude_keys,
            config=self.config,
            # Allow calling init twice if creating more than one trainer in the
            # same process
            reinit=True,
        )

    def __getattr__(self, attr):
        if attr == "metrics":
            return self.optimizer

        raise AttributeError(f"{type(self).__name__} has no '{attr}' attribute")

    @overrd(RLlibTrainer)
    def __getstate__(self):
        state = super().__getstate__()
        state["global_vars"] = self.global_vars
        return state

    @overrd(RLlibTrainer)
    def __setstate__(self, state):
        self.global_vars = state["global_vars"]
        if hasattr(self, "workers"):
            self.workers.foreach_worker(lambda w: w.set_global_vars(self.global_vars))

        super().__setstate__(state)

    @overrd(RLlibTrainer)
    def _stop(self):
        super()._stop()
        if self.config["wandb"]["project"] and wandb:
            wandb.join()
