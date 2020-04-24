"""Module inspection with Streamlit."""
import logging
import sys

from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, DataRange1d
from bokeh.plotting import figure
import pandas as pd
import ray
from ray.rllib.policy.sample_batch import SampleBatch
import streamlit as st

import raylab
from raylab.algorithms.registry import ALGORITHMS
from raylab.utils.checkpoints import get_agent_from_checkpoint

# pylint:disable=invalid-name,missing-docstring,pointless-string-statement
# pylint:disable=no-value-for-parameter
"""
# Visualize Module
"""

logging.getLogger("ray.rllib").setLevel(logging.DEBUG)


@st.cache
def setup():
    ray.init(ignore_reinit_error=True)
    raylab.register_all_agents()
    raylab.register_all_environments()


@st.cache
def load_agent(checkpoint, agent_id, evaluate):
    return get_agent_from_checkpoint(
        checkpoint,
        agent_id,
        use_eval_config=evaluate,
        config_overrides={
            "num_workers": 0,
            "num_envs_per_worker": 1,
            "batch_mode": "complete_episodes",
            "rollout_fragment_length": 1,
        },
    )


def produce_rollout(agent):
    return agent.workers.local_worker().sample()


def scatter_matrix(dataset, target_dataset=None):
    # pylint:disable=too-many-locals
    # https://stackoverflow.com/a/49634658
    dataset_source = ColumnDataSource(data=dataset)

    xdr = DataRange1d(bounds=None)
    ydr = DataRange1d(bounds=None)

    columns = dataset.columns
    if target_dataset is not None:
        assert len(target_dataset.columns) == len(columns)
        target_source = ColumnDataSource(data=target_dataset)

    y_max = len(columns) - 1
    scatter_plots = []
    for yidx, y_col in enumerate(columns):
        scatter_plots += [[]]
        for xidx in range(yidx):
            x_col = columns[xidx]
            yax = xidx == 0
            xax = yidx == y_max
            mbl = 40 if yax else 0
            mbb = 40 if xax else 0

            pic = figure(
                x_range=xdr,
                y_range=ydr,
                x_axis_label=x_col,
                y_axis_label=y_col,
                plot_width=200 + mbl,
                plot_height=200 + mbb,
                min_border_left=2 + mbl,
                min_border_right=2,
                min_border_top=2,
                min_border_bottom=2 + mbb,
            )
            rend = pic.circle(
                source=dataset_source,
                x=x_col,
                y=y_col,
                fill_alpha=0.3,
                line_alpha=0.3,
                size=3,
            )
            # pylint:disable=no-member
            xdr.renderers.append(rend)
            ydr.renderers.append(rend)
            # pylint:enable=no-member
            if target_dataset is not None:
                rend = pic.circle(
                    source=target_source,
                    x=target_dataset.columns[xidx],
                    y=target_dataset.columns[yidx],
                    fill_alpha=0.3,
                    line_alpha=0.3,
                    size=10,
                    color="red",
                )
                # pylint:disable=no-member
                xdr.renderers.append(rend)
                ydr.renderers.append(rend)
                # pylint:enable=no-member

            if not yax:
                pic.yaxis.axis_label = ""
                pic.yaxis.visible = False
            if not xax:
                pic.xaxis.axis_label = ""
                pic.xaxis.visible = False

            scatter_plots[yidx].append(pic)
        scatter_plots[yidx] += [None] * (y_max - yidx)

    return gridplot(
        scatter_plots,
        # ncols=len(columns),
        # sizing_mode="scale_both",
        toolbar_location="right",
    )


# https://discuss.streamlit.io/t/how-can-i-clear-a-specific-cache-only/1963/6
@st.cache(allow_output_mutation=True)
def get_rollout(agent):
    return [produce_rollout(agent)]


def plot_distributions(policy, rollout, timestep):
    obs = policy.convert_to_tensor(rollout[SampleBatch.CUR_OBS][timestep])
    act = policy.convert_to_tensor(rollout[SampleBatch.ACTIONS][timestep])
    new_obs = rollout[SampleBatch.NEXT_OBS][timestep]

    components = []
    if "actor" in policy.module:
        components += ["actor"]
    if "model" in policy.module:
        components += ["model"]

    component = st.selectbox("Inspect module:", options=components)
    if component == "actor":
        plot_actor_distributions(policy.module.actor, obs)
    elif component == "model":
        plot_model_distributions(policy.module.model, obs, act, new_obs)


def plot_actor_distributions(actor, obs):
    if st.checkbox("Deterministic"):
        acts, _ = actor.deterministic(obs)
        acts.unsqueeze_(0)
    else:
        n_samples = st.number_input("Number of action samples", min_value=1, step=1)
        acts, _ = actor.sample(obs, (n_samples,))

    acts = acts.detach().numpy()
    data = {f"act[{i}]": acts[..., i] for i in range(acts.shape[-1])}
    dataset = pd.DataFrame(data)
    chart = scatter_matrix(dataset)
    st.bokeh_chart(chart)


def plot_model_distributions(model, obs, act, new_obs):
    n_samples = st.number_input("Number of observation samples", min_value=1, step=1)
    samples, _ = model.sample(obs, act, (n_samples,))
    samples = samples.numpy()

    data = {f"obs[{i}]": samples[..., i] for i in range(samples.shape[-1])}
    target_data = {f"new_obs[{i}]": new_obs[i][None] for i in range(new_obs.shape[-1])}

    dataset = pd.DataFrame(data)
    target_dataset = pd.DataFrame(target_data)
    chart = scatter_matrix(dataset, target_dataset)
    st.bokeh_chart(chart)


def main():
    setup()
    checkpoint = sys.argv[1]
    options = list(ALGORITHMS.keys()) + [""]
    agent_id = st.selectbox("Algorithm:", options, index=len(options) - 1)

    if agent_id:
        evaluate = st.checkbox("Use evaluation config")
        agent = load_agent(checkpoint, agent_id, evaluate)

        rollout_wrapper = get_rollout(agent)
        if st.button("Sample new episode"):
            rollout_wrapper.clear()
            rollout_wrapper.append(produce_rollout(agent))

        rollout = rollout_wrapper[0]
        timestep = st.slider("Timestep", min_value=0, max_value=rollout.count - 1)

        st.write("Observation", "Action")
        obs = rollout[SampleBatch.CUR_OBS][timestep]
        act = rollout[SampleBatch.ACTIONS][timestep]
        st.write(obs, act)

        plot_distributions(agent.get_policy(), rollout, timestep)


if __name__ == "__main__":
    main()
