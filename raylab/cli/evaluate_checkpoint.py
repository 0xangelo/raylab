"""CLI for rolling out trained policies."""
import itertools

import click

from .utils import initialize_raylab


@click.command()
@click.argument(
    "checkpoint",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
)
@click.option(
    "--agent", required=True, default=None, help="Name of the trainable class to run."
)
@click.option(
    "--env", default=None, help="Name of the environment to interact with, optional."
)
@click.option(
    "--render/--no-render", default=False, help="Whether to render each episode."
)
@click.option(
    "--episodes", "-n", type=float, default=float("inf"), help="Number of episodes."
)
@initialize_raylab
def rollout(checkpoint, agent, env, render, episodes):
    """Simulate an agent from a given checkpoint in the desired environment."""
    # pylint:disable=too-many-locals
    from contextlib import suppress
    import ray
    from raylab.utils.checkpoints import get_agent_from_checkpoint

    ray.init()
    agent = get_agent_from_checkpoint(checkpoint, agent, env)
    env = agent.workers.local_worker().env

    with suppress(KeyboardInterrupt), env:
        for episode in itertools.count():
            if episode >= episodes:
                break

            obs, done, cummulative_reward, time = env.reset(), False, 0, 0
            while not done:
                new_obs, rew, done, _ = env.step(agent.compute_action(obs))
                cummulative_reward += rew
                time += 1
                if render:
                    env.render()
                obs = new_obs

            print("episode length:", time)
            print("cummulative_reward:", cummulative_reward)
