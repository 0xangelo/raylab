LEARNING_STARTS = 10000


def module_config() -> dict:
    from raylab.policy.modules.sac import SACSpec

    spec = SACSpec()
    spec.actor.encoder.units = (256, 256)
    spec.actor.encoder.activation = "ReLU"
    spec.actor.encoder.layer_norm = False
    spec.actor.input_dependent_scale = True
    spec.actor.initial_entropy_coeff = 1.0  # Taken from pfrl
    # spec.actor.initializer = {"name": "xavier_uniform"}

    spec.critic.encoder.units = (256, 256)
    spec.critic.encoder.activation = "ReLU"
    spec.critic.encoder.layer_norm = False
    spec.critic.encoder.delay_action = False
    spec.critic.double_q = True
    spec.critic.parallelize = True
    # spec.critic.initializer = {"name": "xavier_uniform"}

    return {"type": "SAC", **spec.to_dict()}


def policy_config() -> dict:
    from ray import tune

    return {
        "module": module_config(),
        "optimizer": {
            "actor": {"type": "Adam", "lr": 3e-4},
            "critics": {"type": "Adam", "lr": 3e-4},
            "alpha": {"type": "Adam", "lr": 3e-4},  # Taken from pfrl
        },
        "buffer_size": int(1e6),
        "batch_size": 256,
        "improvement_steps": 1,
        "gamma": 0.99,
        "polyak": 0.995,
        # "target_entropy": tune.grid_search(["auto", "tf-agents"]),
        "target_entropy": "auto",
        "exploration_config": {"pure_exploration_steps": LEARNING_STARTS},
    }


def trainer_config(evaluation_interval: int = 10) -> dict:
    return {
        "timesteps_per_iteration": 1000,
        "rollout_fragment_length": 1,
        "evaluation_interval": evaluation_interval,
        "evaluation_num_episodes": 10,
        "evaluation_config": {"explore": False},
        "learning_starts": LEARNING_STARTS,
    }


def env_config(env: str = "HalfCheetah-v3") -> dict:
    conf = {
        "env": env,
        "env_config": {
            "time_aware": False,
            "max_episode_steps": 1000,
            "single_precision": True,
        },
    }
    # if env.endswith("-v3"):
    #     kwargs = dict(exclude_current_positions_from_observation=False)
    #     conf["env_config"]["kwargs"] = kwargs
    return conf


def get_config():
    config = {"policy": policy_config()}
    config.update(trainer_config())
    config.update(env_config())
    return config
