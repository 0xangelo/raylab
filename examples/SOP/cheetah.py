LEARNING_STARTS = 10000


def module_config() -> dict:
    from raylab.policy.modules.ddpg import DDPGSpec

    spec = DDPGSpec()
    spec.actor.network.units = (256, 256)
    spec.actor.network.activation = "ReLU"
    spec.actor.network.layer_norm = False
    spec.actor.network.norm_beta = 1.2
    spec.actor.separate_behavior = False
    spec.actor.smooth_target_policy = True
    spec.actor.target_gaussian_sigma = 0.3
    spec.actor.separate_target_policy = False
    # spec.actor.initializer = {"name": "xavier_uniform"}

    spec.critic.encoder.units = (256, 256)
    spec.critic.encoder.activation = "ReLU"
    spec.critic.encoder.layer_norm = False
    spec.critic.encoder.delay_action = False
    spec.critic.double_q = True
    spec.critic.parallelize = True
    # spec.critic.initializer = {"name": "xavier_uniform"}

    return {"type": "DDPG", **spec.to_dict()}


def policy_config() -> dict:
    return {
        "module": module_config(),
        "optimizer": {
            "actor": {"type": "Adam", "lr": 3e-4},
            "critics": {"type": "Adam", "lr": 3e-4},
        },
        "buffer_size": int(1e6),
        "batch_size": 256,
        "improvement_steps": 1,
        "gamma": 0.99,
        "polyak": 0.995,
        "policy_delay": 1,
        "exploration_config": {
            "type": "raylab.utils.exploration.GaussianNoise",
            "noise_stddev": 0.3,
            "pure_exploration_steps": LEARNING_STARTS,
        },
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
        "env_config": {"time_aware": False, "max_episode_steps": 1000},
    }
    # if env.endswith("-v3"):
    #     conf["env_config"]["exclude_current_positions_from_observation"] = False
    return conf


def get_config():
    config = {"policy": policy_config()}
    config.update(trainer_config())
    config.update(env_config())
    return config
