"""Utilities for incorporating Weights & Biases with Raylab trainers."""
import numbers

from ray.tune.utils import flatten_dict

try:
    import wandb
except ImportError:
    wandb = None


class WandBLogger:
    """Synchronizes training results with Weights&Biases.

    Special configurations include:
    * file_paths: Sequence of file names to pass to `wandb.save` on trainer
      setup.
    * save_checkpoints: Whether to sync trainer checkpoints to W&B via
      `wandb.save`.
    """

    # pylint:disable=missing-docstring

    SPECIAL_KEYS = {"file_paths", "save_checkpoints"}

    def __init__(self, config: dict, name: str):
        create = bool(config.get("wandb", {}))
        self._run = self._setup(config, name) if create else None

    def _setup(self, config: dict, name: str):
        assert wandb is not None, "Unable to import wandb, did you install it via pip?"

        wandb_config = config["wandb"]
        self._save_checkpoints = wandb_config.get("save_checkpoints", False)

        config_exclude_keys = {"wandb", "callbacks"}
        config_exclude_keys.update(set(wandb_config.pop("config_exclude_keys", [])))

        wandb_kwargs = dict(
            name=name,
            config_exclude_keys=config_exclude_keys,
            config=config,
            # Allow calling init twice if creating more than one trainer in the
            # same process
            reinit=True,
        )
        wandb_kwargs.update(
            {k: v for k, v in wandb_config.items() if k not in self.SPECIAL_KEYS}
        )
        run = wandb.init(**wandb_kwargs)

        file_paths = wandb_config.get("file_paths", [])
        for path in file_paths:
            wandb.save(path)

        return run

    @property
    def enabled(self):
        return self._run is not None

    @property
    def run(self):
        return self._run

    @staticmethod
    def log_result(result: dict):
        # Avoid logging the config every iteration
        # Only log Jsonable objects
        tmp = result.copy()
        for k in ["done", "config", "pid", "timestamp"]:
            if k in tmp:
                del tmp[k]
        metrics = {}
        for key, value in flatten_dict(tmp, delimiter="/").items():
            if not isinstance(value, numbers.Number):
                continue
            metrics[key] = value
        wandb.log(metrics)

    def save_checkpoint(self, checkpoint_path: str):
        if self._save_checkpoints:
            wandb.save(checkpoint_path)

    @staticmethod
    def stop():
        wandb.join()
