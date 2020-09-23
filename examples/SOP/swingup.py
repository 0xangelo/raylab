from raylab.cli.utils import tune_experiment


def get_config():
    from ray import tune

    return {
        "env": "CartPoleSwingUp-v1",
        "env_config": {"max_episode_steps": 500, "time_aware": False},
        "dpg_loss": tune.grid_search("default acme".split()),
        "buffer_size": int(1e5),
        "optimizer": {
            "actor": {"type": "Adam", "lr": 3e-4},
            "critics": {"type": "Adam", "lr": 3e-4},
        },
        "polyak": 0.995,
        "module": {
            "type": "DDPG",
            "actor": {
                "smooth_target_policy": True,
                "target_gaussian_sigma": 0.3,
                "beta": 1.2,
                "encoder": {"units": (128, 128), "activation": "Swish"},
            },
            "critic": {
                "double_q": True,
                "encoder": {
                    "units": (128, 128),
                    "activation": "Swish",
                    "delay_action": True,
                },
            },
        },
        "exploration_config": {
            "type": "raylab.utils.exploration.GaussianNoise",
            "noise_stddev": 0.3,
            "pure_exploration_steps": 5000,
        },
        "learning_starts": 5000,
        "rollout_fragment_length": 200,
        "batch_mode": "truncate_episodes",
        "train_batch_size": 128,
        "timesteps_per_iteration": 1000,
        "evaluation_interval": 5,
        "evaluation_config": {
            "env_config": {"max_episode_steps": 1000, "time_aware": False},
        },
    }


@tune_experiment
def main():
    config = get_config()
    return "SOP", config, {"stop": {"timesteps_total": config["buffer_size"]}}


if __name__ == "__main__":
    main()
