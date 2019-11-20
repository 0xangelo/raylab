import itertools

import click
import matplotlib.pyplot as plt
import torch
import numpy as np
import ray
import raylab
from raylab.cli.evaluate_checkpoint import get_agent


def _create_fig():
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.grid()
    return fig, ax


def _render_start_and_goal_positions(ax, start, goal):
    ax.plot(
        [start[0]],
        [start[1]],
        marker="X",
        markersize=15,
        color="limegreen",
        label="initial",
    )
    ax.plot(
        [goal[0]], [goal[1]], marker="X", markersize=15, color="crimson", label="goal"
    )


def _render_deceleration_zones(ax, start, goal, zones, npoints=1000):
    lower = (start[0] - 2.0, start[1] - 2.0)
    upper = (goal[0] + 2.0, goal[1] + 2.0)
    X, Y = np.meshgrid(
        np.linspace(lower[0], upper[0], npoints),
        np.linspace(lower[1], upper[1], npoints),
    )

    Lambda = 1.0
    for (xcenter, ycenter), decay in zones:
        x_diff = np.abs(X - xcenter)
        y_diff = np.abs(Y - ycenter)
        D = np.sqrt(x_diff ** 2 + y_diff ** 2)
        Lambda *= 2 / (1 + np.exp(-decay * D)) - 1.00

    ticks = np.arange(0.0, 1.01, 0.10)
    cp = ax.contourf(X, Y, Lambda, ticks, cmap=plt.cm.bone)
    plt.colorbar(cp, ticks=ticks)
    cp = ax.contour(X, Y, Lambda, ticks, colors="black", linestyles="dashed")


def _render_path(ax, x, y, deltas):
    xdeltas = np.array([d[0] for d in deltas])
    ydeltas = np.array([d[1] for d in deltas])
    ax.streamplot(
        x,
        y,
        np.reshape(xdeltas, (len(x), len(y))),
        np.reshape(ydeltas, (len(x), len(y))),
        density=2.0,
        color="dodgerblue",
    )


#######################################################################################


@click.command()
@click.argument("checkpoint")
@click.option("--algo", default=None, help="Name of the trainable class to run.")
@click.pass_context
def make_streamplot(ctx, checkpoint, algo):
    """Simulate an agent from a given checkpoint in the desired environment."""
    if not algo:
        click.echo("No algorithm name provided, exiting...")
        ctx.exit()

    ray.init()
    raylab.register_all_agents()
    raylab.register_all_environments()

    agent = get_agent(checkpoint, algo, "Navigation")
    env = agent.workers.local_worker().env
    policy = agent.get_policy()

    x = np.linspace(env._start[0] - 1, env._end[0] + 1, 25)
    y = np.linspace(env._start[1] - 1, env._end[1] + 1, 25)

    obs = torch.Tensor(np.stack(list(itertools.product(x, y))))
    obs = torch.cat([obs, torch.zeros_like(obs[..., :1])], dim=-1)

    acts, _, _ = policy.compute_actions(obs, [])

    fig, ax = _create_fig()
    _render_start_and_goal_positions(ax, env._start, env._end)
    _render_deceleration_zones(
        ax, env._start, env._end, zip(env._deceleration_center, env._deceleration_decay)
    )
    _render_path(ax, x, y, acts)
    plt.show()


if __name__ == "__main__":
    make_streamplot()  # pylint: disable=no-value-for-parameter
