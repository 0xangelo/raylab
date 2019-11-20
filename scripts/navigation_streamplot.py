import itertools

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

ray.init()
raylab.register_all_agents()
raylab.register_all_environments()

agent = get_agent(
    # "data/test_policy/checkpoint_100/checkpoint-100", "MAPO", "Navigation"
    # "data/20191119/test6/MAPO-Navigation-Walks/MAPO_Navigation_13_grad_estimator=pathwise_derivative,model_loss=decision_aware,seed=3_2019-11-19_22-47-521vi8s2kw/checkpoint_100/checkpoint-100", "MAPO", "Navigation"
    "data/20191119/test6/SOP-Navigation-Walks/MaryWalks/checkpoint_100/checkpoint-100", "SOP", "Navigation"
)
env = agent.workers.local_worker().env
env_original = env.env

policy = agent.get_policy()
print(env.observation_space.shape)

x = np.linspace(env_original._start[0] - 1, env_original._end[0] + 1, 25)
y = np.linspace(env_original._start[1] - 1, env_original._end[1] + 1, 25)

print(x.shape)
print(y.shape)

obs = torch.Tensor(np.stack(list(itertools.product(x, y))))
obs = torch.cat([obs, 0.2 + torch.zeros((625, 17)) ], dim=-1)


acts, _, _ = policy.compute_actions(obs, [])

fig, ax = _create_fig()
_render_start_and_goal_positions(ax, env_original._start, env_original._end)
_render_deceleration_zones(
    ax, env_original._start, env_original._end, zip(env_original._deceleration_center, env_original._deceleration_decay)
)
_render_path(ax, x, y, acts)
plt.show()
