import csv
import os.path as osp

import ray
import click
import numpy as np

import raylab
from raylab.cli.evaluate_checkpoint import get_agent


@click.command()
@click.argument(
    "checkpoint",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
)
@click.option(
    "--logdir",
    "-l",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True),
)
@click.option(
    "--num_episodes",
    "-n",
    type=int,
    default=1,
    show_default=True,
    help="""Number of episodes to simulate.""",
)
@click.option(
    "--algo", default=None, required=True, help="Name of the trainable class to run."
)
def main(checkpoint, logdir, num_episodes, algo):
    ray.init()
    raylab.register_all_agents()
    raylab.register_all_environments()

    agent = get_agent(checkpoint, algo, "IndustrialBenchmark")
    env = agent.workers.local_worker().env
    fieldnames = [
        k
        for k in env.unwrapped._ib.state.keys()
        if np.isscalar(env.unwrapped._ib.state[k])
    ]
    writer = csv.DictWriter(
        open(osp.join(logdir, "episodes.csv"), "a"),
        fieldnames=fieldnames + ["episode", "time"],
    )
    writer.writeheader()

    def row_dict(episode, time):
        state_dict = {k: env.unwrapped._ib.state[k] for k in fieldnames}
        return {"episode": episode, "time": time, **state_dict}

    with env:
        for episode in range(num_episodes):
            obs, done, time = env.reset(), False, 0
            writer.writerow(row_dict(episode, time))
            while not done:
                obs, rew, done, info = env.step(agent.compute_action(obs))
                time += 1
                writer.writerow(row_dict(episode, time))


if __name__ == "__main__":
    main()
