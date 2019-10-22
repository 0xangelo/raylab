"""Support for updating policies' parameter noise."""
from ray import tune


class ParameterNoiseMixin:
    """Adds method to perturb policies at the beggining of every episode."""

    # pylint: disable=too-few-public-methods

    @staticmethod
    def _set_parameter_noise_callbacks(config):
        # Taken from ray.rllib.agents.dqn.dqn
        if config["exploration"] == "parameter_noise":
            if config["batch_mode"] != "complete_episodes":
                raise ValueError(
                    "Exploration with parameter space noise requires "
                    "batch_mode to be complete_episodes."
                )

            if config["callbacks"]["on_episode_start"]:
                start_callback = config["callbacks"]["on_episode_start"]
            else:
                start_callback = None

            def on_episode_start(info):
                # as a callback function to sample and pose parameter space
                # noise on the parameters of network
                policies = info["policy"]
                for pol in policies.values():
                    pol.perturb_policy_parameters()
                if start_callback:
                    start_callback(info)

            config["callbacks"]["on_episode_start"] = tune.function(on_episode_start)
