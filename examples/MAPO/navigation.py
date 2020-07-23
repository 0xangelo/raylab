import click

from raylab.cli.utils import tune_options


def get_base_config():
    return {
        # === SACTorchPolicy ===
        "target_entropy": "auto",
        # === TargetNetworksMixin ===
        "polyak": 0.995,
        # === ModelTrainingMixin ===
        "model_training": {
            "dataloader": {"batch_size": 128, "replacement": True},
            "max_epochs": 2,
            "max_grad_steps": 120,
            "max_time": 5,
            "improvement_threshold": None,
        },
        # === Policy ===
        "module": {
            "type": "ModelBasedSAC",
            "model": {
                "ensemble_size": 1,
                "residual": True,
                "input_dependent_scale": True,
                "network": {"units": (128, 128), "activation": "Swish"},
            },
            "actor": {
                "encoder": {"units": (128, 128), "activation": "Swish"},
                "input_dependent_scale": True,
                "initial_entropy_coeff": 0.05,
            },
            "critic": {
                "encoder": {"units": (128, 128), "activation": "Swish"},
                "double_q": True,
            },
        },
        # PyTorch optimizers to use
        "torch_optimizer": {
            "models": {"type": "Adam", "lr": 3e-4},
            "actor": {"type": "Adam", "lr": 3e-4},
            "critics": {"type": "Adam", "lr": 3e-4},
            "alpha": {"type": "Adam", "lr": 3e-4},
        },
        "exploration_config": {"pure_exploration_steps": 500},
        # === ModelBasedTrainer ===
        "holdout_ratio": 0,
        "max_holdout": 0,
        # === OffPolicyTrainer ===
        "policy_improvements": 10,
        "buffer_size": int(1e4),
        "learning_starts": 500,
        # === Trainer ===
        "timesteps_per_iteration": 200,
        "evaluation_interval": 5,
        # === Rollout Worker ===
        "rollout_fragment_length": 20,
        "batch_mode": "truncate_episodes",
    }


def register_custom_env():
    from ray.tune import register_env
    from raylab.envs import (
        register_reward_fn,
        register_termination_fn,
        get_env_creator,
        get_reward_fn,
        get_termination_fn,
    )
    from raylab.envs.wrappers import RandomIrrelevant as wrapper_cls

    def env_maker(env_config):
        base = get_env_creator("Navigation")(env_config)
        wrapped = wrapper_cls(base, size=env_config["irrelevant_size"])
        return wrapped

    def reward_fn(env_config):
        base = get_reward_fn("Navigation", env_config)
        irrelevant_size = env_config["irrelevant_size"]
        return wrapper_cls.wrap_env_function(base, irrelevant_size)

    def termination_fn(env_config):
        base = get_termination_fn("Navigation", env_config)
        irrelevant_size = env_config["irrelevant_size"]
        return wrapper_cls.wrap_env_function(base, irrelevant_size)

    env_id = "NavigationRI"
    register_env(env_id, env_maker)
    register_reward_fn(env_id)(reward_fn)
    register_termination_fn(env_id)(termination_fn)

    return env_id


def common_setup(tune_kwargs):
    import ray
    from ray import tune
    import raylab

    ray.init()
    raylab.register_all()

    config = get_base_config()
    config["env"] = register_custom_env()
    config["env_config"] = {"irrelevant_size": 16}
    config["wandb"] = {"project": "mapo-icaps", "entity": "reinforcers"}

    overrides = dict(
        # Save weights while evaluating target policy
        checkpoint_freq=config["evaluation_interval"],
        # Use buffer size as a proxy for total number of timesteps
        stop=dict(timesteps_total=config["buffer_size"]),
    )
    tune_kwargs.update(overrides)

    return config, tune_kwargs


@click.group()
def main():
    pass


@main.command()
@tune_options
def mapo(tune_kwargs):
    from ray import tune

    trainable = "MAPO"
    config, tune_kwargs = common_setup(tune_kwargs)
    config["losses"] = {"grad_estimator": "PD", "lambda": 0.01, "model_samples": 1}

    tune.run(trainable, config=config, **tune_kwargs)


@main.command()
@tune_options
def mapo_mle(tune_kwargs):
    from ray import tune

    trainable = "MAPO-MLE"
    config, tune_kwargs = common_setup(tune_kwargs)
    config["losses"] = {"grad_estimator": "PD", "model_samples": 1}

    tune.run(trainable, config=config, **tune_kwargs)


if __name__ == "__main__":
    main()
