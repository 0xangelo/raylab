"""Primitives for all Trainers."""
from abc import ABCMeta
from typing import Callable
from typing import Optional

from ray.rllib import Policy
from ray.rllib.agents.trainer import Trainer as RLlibTrainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils import override as overrides
from ray.tune import Trainable

from raylab.utils.wandb import WandBLogger

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

    A WorkerSet must be set (as a `workers` attribute) to collect episode
    statistics.

    Always creates a `StandardMetrics` instance as the `metrics` attribute to
    log episode metrics (to be removed in the future).

    Integration with `Weights & Biases`_ is available. The user must install
    `wandb` and set a project name in the `wandb` subconfig dict.

    .. _`Weights & Biases`: https://docs.wandb.com/
    """

    # pylint:disable=too-many-instance-attributes
    workers: Optional[WorkerSet]
    metrics: Optional[compat.StandardMetrics]
    wandb: WandBLogger
    _name: str = ""
    _policy: Optional[Policy] = None
    # Handle all config merging in RaylabOptions
    options: RaylabOptions = RaylabOptions()

    @overrides(RLlibTrainer)
    def train(self):
        result = {}

        # Run evaluation once before any optimization is done
        if self.iteration == 0 and self.config["evaluation_interval"]:
            result.update(self._evaluate())

        result.update(super().train())

        if self.wandb.enabled:
            self.wandb.log_result(result)

        return result

    @property
    @overrides(RLlibTrainer)
    def _default_config(self):
        return self.options.defaults

    @overrides(RLlibTrainer)
    def setup(self, config: dict):
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

        if hasattr(self, "workers"):
            self.metrics = compat.StandardMetrics(self.workers)

        # Evaluation setup.
        if config.get("evaluation_interval"):
            evaluation_config = compat.setup_evaluation_config(config)
            self.evaluation_workers = self._make_workers(
                self.env_creator,
                self._policy,
                evaluation_config,
                num_workers=config["evaluation_num_workers"],
            )

        if self.config["compile_policy"]:
            self._optimize_policy_backend()

        self.wandb = WandBLogger(self.config, self._name)

    def _optimize_policy_backend(self):
        if not hasattr(self, "workers"):
            raise RuntimeError(
                f"{type(self).__name__} has no worker set. "
                "Cannot access policies for compilation."
            )
        self.workers.foreach_policy(lambda p, _: p.compile())

    @overrides(RLlibTrainer)
    def collect_metrics(self, selected_workers: Optional[list] = None) -> dict:
        """Collects metrics from the remote workers of this agent."""
        return self.metrics.collect_metrics(
            self.config["collect_metrics_timeout"],
            min_history=self.config["metrics_smoothing_episodes"],
            selected_workers=selected_workers,
        )

    @classmethod
    @overrides(RLlibTrainer)
    def default_resource_request(cls, config: dict) -> compat.Resources:
        return compat.default_resource_request(cls, config)

    @overrides(Trainable)
    def save(self, checkpoint_dir=None):
        checkpoint_path = super().save(checkpoint_dir)
        if self.wandb.enabled:
            self.wandb.save_checkpoint(checkpoint_path)
        return checkpoint_path

    @overrides(RLlibTrainer)
    def __getstate__(self):
        state = super().__getstate__()
        state["global_vars"] = self.global_vars

        if hasattr(self, "metrics"):
            state["metrics"] = self.metrics.save()

        return state

    @overrides(RLlibTrainer)
    def __setstate__(self, state):
        self.global_vars = state["global_vars"]
        if hasattr(self, "workers"):
            self.workers.foreach_worker(lambda w: w.set_global_vars(self.global_vars))

        if hasattr(self, "metrics"):
            self.metrics.restore(state["metrics"])

        super().__setstate__(state)

    @overrides(RLlibTrainer)
    def cleanup(self):
        super().cleanup()
        if self.wandb.enabled:
            self.wandb.stop()
