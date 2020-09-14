"""Primitives for all Trainers."""
from abc import ABCMeta
from typing import Optional

from ray.rllib import Policy
from ray.rllib.agents import Trainer as RLlibTrainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils import override as overrides
from ray.tune import Trainable

from raylab.options import configure
from raylab.options import option
from raylab.options import TrainerOptions
from raylab.utils.wandb import WandBLogger

from . import compat


# ==============================================================================
# Base Raylab Trainer
# ==============================================================================
@configure
@option(
    "policy/",
    allow_unknown_subkeys=True,
    help="""Sub-configurations for the policy class.""",
)
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
@option("framework", default="torch", override=True)
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
    options: TrainerOptions = TrainerOptions()

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
                " Did you call `configure` as the last decorator?"
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

        if self.config["policy"].get("compile", False):
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
