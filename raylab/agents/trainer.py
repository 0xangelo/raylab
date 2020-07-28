"""Primitives for all Trainers."""
import warnings
from abc import ABCMeta
from typing import Callable
from typing import Optional

try:
    import wandb
except ImportError:
    wandb = None

from ray.rllib.agents.trainer import Trainer as RLlibTrainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.optimizers import PolicyOptimizer
from ray.rllib.utils import override as overrides
from ray.tune import Trainable

from . import compat
from .options import Json
from .options import RaylabOptions


# ==============================================================================
# Programatic config setting
# ==============================================================================
def configure(cls: type) -> type:
    """Decorator for finishing the configuration setup for a Trainer class.

    Should be called after all :func:`config` decorators have been applied,
    i.e., as the top-most decorator.
    """
    cls.options = cls.options.copy_and_set_queued_options()
    return cls


def option(
    key: str,
    default: Json = None,
    *,
    help: Optional[str] = None,
    override: bool = False,
    allow_unknown_subkeys: bool = False,
    override_all_if_type_changes: bool = False,
    separator: str = "/",
) -> Callable[[type], type]:
    """Returns a decorator for adding/overriding a Trainer class config.

    If `key` ends in a separator and `default` is None, treats the option as a
    nested dict of options and sets the default to an empty dictionary.

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
    """
    # pylint:disable=too-many-arguments,redefined-builtin
    def _queue(cls):
        cls.options.add_option_to_queue(
            key=key,
            default=default,
            info=help,
            override=override,
            allow_unknown_subkeys=allow_unknown_subkeys,
            override_all_if_type_changes=override_all_if_type_changes,
            separator=separator,
        )
        return cls

    return _queue


# ==============================================================================
# Base Raylab Trainer
# ==============================================================================
Jsonable = (dict, list, str, int, float, bool, type(None))


@configure
@option(
    "wandb/",
    allow_unknown_subkeys=True,
    help="""Configs for integration with Weights & Biases.

    Accepts arbitrary keyword arguments to pass to `wandb.init`.
    The defaults for `wandb.init` are:
    * name: `_name` property of the trainer.
    * config: full `config` attribute of the trainer
    * config_exclude_keys: `wandb` and `callbacks` configs
    * reinit: True

    Don't forget to:
      * install `wandb` via pip
      * login to W&B with the appropriate API key for your
        team/project.
      * set the `wandb/project` name in the config dict

    Check out the Quickstart for more information:
    `https://docs.wandb.com/quickstart`
    """,
)
@option(
    "wandb/enabled",
    default=False,
    help="""Whether to sync logs and files with `wandb`.""",
)
@option(
    "wandb/file_paths",
    default=(),
    help="Sequence of file names to pass to `wandb.save` on trainer setup.",
)
@option(
    "wandb/save_checkpoints",
    default=False,
    help="Whether to sync trainer checkpoints to W&B via `wandb.save`.",
)
@option("compile_policy", False, help="Whether to optimize the policy's backend")
@option(
    "module/",
    help="Type and config of the PyTorch NN module.",
    allow_unknown_subkeys=True,
    override_all_if_type_changes=True,
)
@option(
    "torch_optimizer/",
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
    # Handle all config merging in RaylabOptions
    options: RaylabOptions = RaylabOptions()

    @overrides(RLlibTrainer)
    def train(self):
        result = {}

        # Run evaluation once before any optimization is done
        if self._iteration == 0 and self.config["evaluation_interval"]:
            result.update(self._evaluate())

        result.update(super().train())

        # Update global_vars after training so that they're saved if checkpointing
        self.global_vars["timestep"] = self.metrics.num_steps_sampled

        if self.config["wandb"]["enabled"]:
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

    @property
    @overrides(RLlibTrainer)
    def _default_config(self):
        return self.options.defaults

    @overrides(RLlibTrainer)
    def _setup(self, config: dict):
        if not self.options.all_options_set:
            raise RuntimeError(
                f"{type(self).__name__} still has configs to be set."
                " Did you call `trainer.configure` as the last decorator?"
            )

        self.config = config = self.options.merge_defaults_with(config)

        self.env_creator = compat.make_env_creator(self._env_id, config)

        compat.check_and_resolve_framework_settings(config)

        RLlibTrainer._validate_config(config)

        self.callbacks = compat.validate_callbacks(config)

        compat.set_rllib_log_level(config)

        self._init(config, self.env_creator)

        # Evaluation setup.
        if config.get("evaluation_interval"):
            evaluation_config = compat.setup_evaluation_config(config)
            self.evaluation_workers = self._make_workers(
                self.env_creator,
                self._policy,
                evaluation_config,
                num_workers=config["evaluation_num_workers"],
            )

        self._setup_optimizer_placeholder()

        if self.config["compile_policy"]:
            self._optimize_policy_backend()

        if self.config["wandb"]["enabled"]:
            self._setup_wandb()

    def _setup_optimizer_placeholder(self):
        # Always have a PolicyOptimizer if possible to collect metrics
        if not hasattr(self, "optimizer"):
            if hasattr(self, "workers"):
                self.optimizer = PolicyOptimizer(self.workers)
            else:
                warnings.warn(
                    "No worker set initialized; episodes summary will be unavailable."
                )
        # Always have a WorkerSet if possible to get workers and policy
        if hasattr(self, "optimizer") and not hasattr(self, "workers"):
            self.workers = self.optimizer.workers

    def _optimize_policy_backend(self):
        if not hasattr(self, "workers"):
            raise RuntimeError(
                f"{type(self).__name__} has no worker set. "
                "Cannot access policies for compilation."
            )
        self.workers.foreach_policy(lambda p, _: p.compile())

    def _setup_wandb(self):
        assert wandb is not None, "Unable to import wandb, did you install it via pip?"

        wandb_kwargs = dict(
            name=self._name,
            config_exclude_keys={"wandb", "callbacks"},
            config=self.config,
            # Allow calling init twice if creating more than one trainer in the
            # same process
            reinit=True,
        )
        special_keys = {"enabled", "file_paths", "save_checkpoints"}
        wandb_kwargs.update(
            {k: v for k, v in self.config["wandb"].items() if k not in special_keys}
        )
        wandb.init(**wandb_kwargs)

        file_paths = self.config["wandb"]["file_paths"]
        for path in file_paths:
            wandb.save(path)

    def __getattr__(self, attr):
        if attr == "metrics":
            return self.optimizer

        raise AttributeError(f"{type(self).__name__} has no '{attr}' attribute")

    @classmethod
    @overrides(RLlibTrainer)
    def default_resource_request(cls, config: dict) -> compat.Resources:
        return compat.default_resource_request(cls, config)

    @overrides(Trainable)
    def save(self, checkpoint_dir=None):
        checkpoint_path = super().save(checkpoint_dir)

        config = self.config
        if config["wandb"]["enabled"] and config["wandb"]["save_checkpoints"]:
            wandb.save(checkpoint_path)

        return checkpoint_path

    @overrides(RLlibTrainer)
    def __getstate__(self):
        state = super().__getstate__()
        state["global_vars"] = self.global_vars
        return state

    @overrides(RLlibTrainer)
    def __setstate__(self, state):
        self.global_vars = state["global_vars"]
        if hasattr(self, "workers"):
            self.workers.foreach_worker(lambda w: w.set_global_vars(self.global_vars))

        super().__setstate__(state)

    @overrides(RLlibTrainer)
    def _stop(self):
        super()._stop()
        if self.config["wandb"]["enabled"] and wandb:
            wandb.join()
