from raylab.cli.utils import tune_experiment


def policy_config():
    return {
        # === Off Policy ===
        "buffer_size": int(1e5),
        "batch_size": 128,
        "std_obs": False,
        "target_entropy": "auto",
        "optimizer": {
            "actor": {"type": "Adam", "lr": 3e-4},
            "critics": {"type": "Adam", "lr": 3e-4},
            "alpha": {"type": "Adam", "lr": 3e-4},
        },
        "module": {
            "actor": {"encoder": {"units": (128, 128), "activation": "ReLU"}},
            "critic": {"encoder": {"units": (128, 128), "activation": "ReLU"}},
        },
        "exploration_config": {"pure_exploration_steps": 5000},
    }


def env_config():
    return {
        "env": "CartPoleSwingUp-v1",
        "env_config": {
            "max_episode_steps": 500,
            "time_aware": False,
            "single_precision": False,
        },
    }


def trainer_config():
    return {
        "rollout_fragment_length": 1,
        "batch_mode": "truncate_episodes",
        "timesteps_per_iteration": 1000,
        "learning_starts": 5000,
        "evaluation_interval": 5,
        "evaluation_config": {
            "env_config": {"max_episode_steps": 1000, "time_aware": False},
        },
    }


def wandb_config():
    return {"project": "baselines", "entity": "angelovtt"}


def get_config():
    cnf = {}
    cnf.update(env_config())
    cnf.update({"policy": policy_config()})
    cnf.update({"wandb": wandb_config()})
    cnf.update(trainer_config())
    return cnf


@tune_experiment
def main():
    config = get_config()
    tune_kwargs = dict(stop={"timesteps_total": config["policy"]["buffer_size"]})
    return "SoftAC", config, tune_kwargs


if __name__ == "__main__":
    main()
