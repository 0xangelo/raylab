# pylint:disable=missing-docstring
import click

from raylab.cli.utils import initialize_raylab


class IBEpisodeLogger:
    # pylint:disable=too-few-public-methods
    def __init__(self, env, logdir):
        # pylint:disable=protected-access
        import csv
        import os
        import os.path as osp

        import numpy as np

        self._ib = env.unwrapped._ib
        fieldnames = [
            k for k in self._ib.state.keys() if np.isscalar(self._ib.state[k])
        ]
        if not osp.exists(logdir):
            os.makedirs(logdir)
        self.writer = csv.DictWriter(
            open(osp.join(logdir, "episodes.csv"), "a"),
            fieldnames=fieldnames + ["episode", "time"],
        )
        self.writer.writeheader()

    def write(self, episode, rew, time):
        row = self._row_dict(episode, rew, time)
        self.writer.writerow(row)

    def _row_dict(self, episode, rew, time):
        state_dict = {k: self._ib.state[k] for k in self.writer.fieldnames}
        return {"episode": episode, "reward": rew, "time": time, **state_dict}


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
    "--num-episodes",
    "-n",
    type=int,
    default=1,
    show_default=True,
    help="""Number of episodes to simulate.""",
)
@click.option(
    "--algo", default=None, required=True, help="Name of the trainable class to run."
)
@initialize_raylab
def main(checkpoint, logdir, num_episodes, algo):
    # pylint:disable=too-many-locals
    import ray

    from raylab.utils.experiments import get_agent

    ray.init()
    agent = get_agent(checkpoint, algo, "IndustrialBenchmark")
    env = agent.workers.local_worker().env
    logger = IBEpisodeLogger(env, logdir)

    with env:
        for episode in range(num_episodes):
            obs, rew, time, done = env.reset(), 0, 0, False
            logger.write(episode, rew, time)
            while not done:
                obs, rew, done, _ = env.step(agent.compute_action(obs))
                time += 1
                logger.write(episode, rew, time)


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
