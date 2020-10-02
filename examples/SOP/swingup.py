from raylab.cli.utils import tune_experiment


def policy_config():
    return {
        "dpg_loss": "default",
        "buffer_size": int(1e5),
        "batch_size": 128,
        "optimizer": {
            "actor": {"type": "Adam", "lr": 3e-4},
            "critics": {"type": "Adam", "lr": 3e-4},
        },
        "polyak": 0.995,
        "module": {
            "type": "SOP",
            "actor": {
                "target_gaussian_sigma": 0.3,
                "beta": 1.2,
                "encoder": {"units": (128, 128), "activation": "ReLU"},
            },
            "critic": {
                "double_q": True,
                "encoder": {
                    "units": (128, 128),
                    "activation": "ReLU",
                    "delay_action": False,
                },
            },
        },
        "exploration_config": {
            "type": "raylab.utils.exploration.GaussianNoise",
            "noise_stddev": 0.3,
            "pure_exploration_steps": 5000,
        },
    }


def get_config():
    return {
        "env": "CartPoleSwingUp-v1",
        "env_config": {"max_episode_steps": 500, "time_aware": False},
        "policy": policy_config(),
        "learning_starts": 5000,
        "rollout_fragment_length": 1,
        "batch_mode": "truncate_episodes",
        "timesteps_per_iteration": 1000,
        "evaluation_interval": 5,
        "evaluation_config": {
            "env_config": {"max_episode_steps": 1000, "time_aware": False},
        },
    }


@tune_experiment
def main():
    config = get_config()
    tune_kwargs = dict(stop={"timesteps_total": config["buffer_size"]})
    return "SOP", config, tune_kwargs


if __name__ == "__main__":
    main()
