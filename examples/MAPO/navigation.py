import click

from raylab.cli.utils import tune_options


# ==============================================================================
# Configurations for algorithms and networks. One can import Tune to perform
# grid searches and the like.
# ==============================================================================


def model_free_config():
    """Configurations for 'sample-efficient SAC'"""
    return {
        # === SACTorchPolicy ===
        "target_entropy": "auto",
        "polyak": 0.995,
        # === TorchPolicy ===
        "module": model_free_network_config(),
        "torch_optimizer": {
            "actor": {"type": "Adam", "lr": 3e-4, "weight_decay": 1e-4},
            "critics": {"type": "Adam", "lr": 3e-4, "weight_decay": 1e-4},
            "alpha": {"type": "Adam", "lr": 3e-4},
        },
        "exploration_config": {"pure_exploration_steps": 500},
        # === OffPolicyTrainer ===
        "buffer_size": int(1e4),
        "policy_improvements": 10,
        "learning_starts": 500,
        "train_batch_size": 64,
        # === Trainer ===
        "evaluation_interval": 5,
        # === Rollout Worker ===
        "rollout_fragment_length": 20,
        "batch_mode": "truncate_episodes",
    }


def model_free_network_config():
    """Check `raylab.policy.modules.sac`."""
    return {
        "type": "SAC",
        "actor": {
            "encoder": {"units": (128, 128), "activation": "Swish"},
            "input_dependent_scale": True,
            "initial_entropy_coeff": 0.05,
        },
        "critic": {
            "encoder": {"units": (128, 128), "activation": "Swish"},
            "double_q": True,
        },
    }


def model_based_config():
    """Shared configuration between MAPO and MAPO-MLE"""
    return {
        # === SACTorchPolicy ===
        "target_entropy": "auto",
        "polyak": 0.995,
        # === ModelTrainingMixin ===
        "model_training": {
            "dataloader": {"batch_size": 64, "replacement": True},
            "max_epochs": 2,
            "max_grad_steps": 120,
            "max_time": 5,
            "improvement_threshold": None,
        },
        # === TorchPolicy ===
        "module": model_based_network_config(),
        "torch_optimizer": {
            "models": {"type": "Adam", "lr": 3e-4, "weight_decay": 1e-4},
            "actor": {"type": "Adam", "lr": 3e-4, "weight_decay": 1e-4},
            "critics": {"type": "Adam", "lr": 3e-4, "weight_decay": 1e-4},
            "alpha": {"type": "Adam", "lr": 3e-4},
        },
        "exploration_config": {"pure_exploration_steps": 500},
        # === ModelBasedTrainer ===
        "timesteps_per_iteration": 200,
        "policy_improvements": 10,
        "train_batch_size": 64,
        "holdout_ratio": 0,
        "max_holdout": 0,
        # === OffPolicyTrainer ===
        "buffer_size": int(1e4),
        "learning_starts": 500,
        # === Trainer ===
        "evaluation_interval": 5,
        # === Rollout Worker ===
        "rollout_fragment_length": 20,
        "batch_mode": "truncate_episodes",
    }


def model_based_network_config():
    """Check `raylab.policy.modules.model_based_sac`.

    Might wanna reproduce experiments with bottlenecked network.
    """
    return {
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
    }


# ==============================================================================
# Uncomment these functions to test different environments
# ==============================================================================

# def register_custom_env():
#     from ray.tune import register_env

#     def env_maker(env_config):
#         from raylab.envs import get_env_creator
#         from raylab.envs.wrappers import RandomIrrelevant as wrapper_cls

#         base = get_env_creator("Navigation")(env_config)
#         wrapped = wrapper_cls(base, size=env_config["irrelevant_size"])
#         return wrapped

#     env_id = "NavigationRI"
#     register_env(env_id, env_maker)
# return {"env": env_id, "env_config": {"irrelevant_size": 16}}


# def register_custom_env():
#     from ray.tune import register_env

#     def env_maker(env_config):
#         from raylab.envs import get_env_creator
#         from raylab.envs.wrappers import LinearRedundant as wrapper_cls

#         base = get_env_creator("Navigation")(env_config)
#         wrapped = wrapper_cls(base)
#         return wrapped

#     env_id = "NavigationLR"
#     register_env(env_id, env_maker)
# return {"env": env_id}


def register_custom_env():
    from ray.tune import register_env

    def env_maker(env_config):
        from raylab.envs import get_env_creator
        from raylab.envs.wrappers import CorrelatedIrrelevant as wrapper_cls

        base = get_env_creator("Navigation")(env_config)
        wrapped = wrapper_cls(base, size=env_config["irrelevant_size"])
        return wrapped

    env_id = "NavigationCI"
    register_env(env_id, env_maker)
    return {"env": env_id, "env_config": {"irrelevant_size": 16}}


# def register_custom_env():
#     from ray.tune import register_env

#     def env_maker(env_config):
#         from raylab.envs import get_env_creator
#         from raylab.envs.wrappers import CorrelatedIrrelevant as wrapper_cls

#         base = get_env_creator("Reservoir")(env_config)
#         wrapped = wrapper_cls(base, size=env_config["irrelevant_size"])
#         return wrapped

#     env_id = "ReservoirCI"
#     register_env(env_id, env_maker)
# return {"env": env_id, "env_config": {"irrelevant_size": 16}}


# def register_custom_env():
#     from ray.tune import register_env

#     def env_maker(env_config):
#         from raylab.envs import get_env_creator
#         from raylab.envs.wrappers import NonlinearRedundant as wrapper_cls

#         base = get_env_creator("Reservoir")(env_config)
#         wrapped = wrapper_cls(base)
#         return wrapped

#     env_id = "ReservoirNR"
#     register_env(env_id, env_maker)
#     return {"env": env_id, "env_config": {"irrelevant_size": 16}}


# ==============================================================================
# Common setups
# ==============================================================================


def wandb_config():
    return {"wandb": {"project": "mapo-icaps", "entity": "reinforcers"}}


def common_setup(tune_kwargs, config_fn):
    import ray
    from ray import tune
    import raylab

    ray.init()
    raylab.register_all()

    config = config_fn()
    config.update(register_custom_env())
    config.update(wandb_config())

    overrides = dict(
        # Save weights while evaluating target policy
        checkpoint_freq=config["evaluation_interval"],
        # Use buffer size as a proxy for total number of timesteps
        stop=dict(timesteps_total=config["buffer_size"]),
    )
    tune_kwargs.update(overrides)

    return config, tune_kwargs


# ==============================================================================
# Command Line Interface
# ==============================================================================


@click.group()
def main():
    pass


@main.command()
@tune_options
def mapo(tune_kwargs):
    from ray import tune

    trainable = "MAPO"
    config, tune_kwargs = common_setup(tune_kwargs, model_based_config)
    config["losses"] = {
        "grad_estimator": tune.grid_search(["PD", "SF"]),
        "lambda": 0.01,
        "model_samples": 1,
    }

    tune.run(trainable, config=config, **tune_kwargs)


@main.command()
@tune_options
def mapo_mle(tune_kwargs):
    from ray import tune

    trainable = "MAPO-MLE"
    config, tune_kwargs = common_setup(tune_kwargs, model_based_config)
    config["losses"] = {
        "grad_estimator": tune.grid_search(["PD", "SF"]),
        "model_samples": 1,
    }

    tune.run(trainable, config=config, **tune_kwargs)


@main.command()
@tune_options
def se_sac(tune_kwargs):
    from ray import tune

    trainable = "SoftAC"
    config, tune_kwargs = common_setup(tune_kwargs, model_free_config)

    tune.run(trainable, config=config, **tune_kwargs)


@main.command()
@tune_options
def debug(tune_kwargs):
    from raylab.envs import get_env_creator

    config, _ = common_setup(tune_kwargs)
    env = get_env_creator(config["env"])(config["env_config"])
    print(env)
    print(type(env))
    print(dir(env))


if __name__ == "__main__":
    main()
