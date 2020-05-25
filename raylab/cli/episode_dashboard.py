"""Episode monitoring with Streamlit."""
import os.path as osp
import shelve

import numpy as np
import pandas as pd
import streamlit as st
from bokeh.layouts import gridplot
from bokeh.plotting import figure

# pylint:disable=invalid-name,missing-docstring,pointless-string-statement
# pylint:disable=no-value-for-parameter
"""
# Visualize Episode
"""


def main():
    # pylint:disable=too-many-locals
    import sys

    assert len(sys.argv) <= 2, "Only one episode log from `rllib rollout` allowed."
    path, _ = osp.splitext(sys.argv[1])

    episodes = []
    with shelve.open(path) as rollouts:
        for episode_index in range(rollouts["num_episodes"]):
            episodes += [rollouts[str(episode_index)]]

    rows = []
    for idx, episode in enumerate(episodes):
        for timestep, transition in enumerate(episode):
            obs, act, _, rew, done, *_ = transition
            row = {"episode": idx, "timestep": timestep, "reward": rew, "done": done}
            for dim, ob_ in enumerate(obs):
                row[f"obs[{dim}]"] = ob_
            for dim, ac_ in enumerate(act):
                row[f"act[{dim}]"] = ac_
            rows += [row]

    keys = rows[0].keys()
    data = pd.DataFrame(data={k: np.stack([r[k] for r in rows]) for k in keys})
    data["done"] = data["done"].astype(np.float32)

    stats_columns = [c for c in data.columns if c not in {"timestep", "episode"}]
    grouped = data.groupby("timestep")[stats_columns]
    means = grouped.mean()
    stds = grouped.std()

    pics = []
    for key in stats_columns:
        pic = figure(title=key)
        pic.xaxis.axis_label = "timestep"
        pic.yaxis.axis_label = "value"

        # pylint:disable=too-many-function-args
        pic.line(means.index, means[key])
        pic.varea(
            means.index,
            y1=means[key] - stds[key],
            y2=means[key] + stds[key],
            fill_alpha=0.25,
        )
        # pylint:enable=too-many-function-args

        pics += [[pic]]

    st.bokeh_chart(
        gridplot(
            pics,
            plot_width=350,
            plot_height=100,
            sizing_mode="scale_both",
            toolbar_location="right",
        )
    )


if __name__ == "__main__":
    main()
