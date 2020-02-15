"""CLI for rolling out trained policies."""
import click

from .utils import initialize_raylab


@click.command()
@click.argument("checkpoint")
@click.option(
    "--algo", required=True, default=None, help="Name of the trainable class to run."
)
@click.option(
    "--env", default=None, help="Name of the environment to interact with, optional."
)
@initialize_raylab
def rollout(checkpoint, algo, env):
    """Simulate an agent from a given checkpoint in the desired environment."""
    from contextlib import suppress
    import ray
    from raylab.utils.experiments import get_agent

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
