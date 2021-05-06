LEARNING_STARTS = 1000


def policy_config() -> dict:
    return {
        "buffer_size": int(1e5),
        "exploration_config": {"pure_exploration_steps": LEARNING_STARTS},
    }


def env_config(env: str = "Pusher-v2") -> dict:
    return {
        "env": env,
        "env_config": {
            "time_aware": True,
            "max_episode_steps": 100,
            "single_precision": True,
        },
    }


def trainer_config() -> dict:
    return {
        "timesteps_per_iteration": 1000,
        "rollout_fragment_length": 1,
        "evaluation_interval": 1,
        "evaluation_num_episodes": 10,
        "evaluation_config": {"explore": False},
        "learning_starts": LEARNING_STARTS,
    }


def get_config() -> dict:
    cnf = {"policy": policy_config()}
    cnf.update(env_config())
    cnf.update(trainer_config())
    return cnf
