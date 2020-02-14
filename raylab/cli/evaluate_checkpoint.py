"""CLI for rolling out trained policies."""
import click

from .utils import initialize_raylab


def get_agent(checkpoint, algo, env):
    """Instatiate and restore agent class from checkpoint."""
    import os
    import pickle
    from ray.tune.registry import TRAINABLE_CLASS, _global_registry
    from ray.rllib.utils import merge_dicts

    config = {}
    # Load configuration from file
    config_dir = os.path.dirname(checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    if not os.path.exists(config_path):
        raise ValueError(
            "Could not find params.pkl in either the checkpoint dir or "
            "its parent directory."
        )
    with open(config_path, "rb") as file:
        config = pickle.load(file)

    if "evaluation_config" in config:
        eval_conf = config["evaluation_config"]
        config = merge_dicts(config, eval_conf)

    agent_cls = _global_registry.get(TRAINABLE_CLASS, algo)
    agent = agent_cls(env=env, config=config)
    agent.restore(checkpoint)
    return agent


@click.command()
@click.argument("checkpoint")
@click.option("--algo", default=None, help="Name of the trainable class to run.")
@click.option(
    "--env", default=None, help="Name of the environment to interact with, optional."
)
@click.pass_context
@initialize_raylab
def rollout(ctx, checkpoint, algo, env):
    """Simulate an agent from a given checkpoint in the desired environment."""
    from contextlib import suppress

    import ray

    if not algo:
        ctx.exit()

    ray.init()
    agent = get_agent(checkpoint, algo, env)
    env = agent.workers.local_worker().env

    horizon = agent.config["horizon"] or float("inf")
    obs, done, cummulative_reward, time = env.reset(), True, 0, 0
    with suppress(KeyboardInterrupt), env:
        while True:
            obs, rew, done, _ = env.step(agent.compute_action(obs))
            cummulative_reward += rew
            time += 1
            env.render()
            if done or time >= horizon:
                print("episode length:", time)
                print("cummulative_reward:", cummulative_reward)
                obs, done, cummulative_reward, time = env.reset(), False, 0, 0
