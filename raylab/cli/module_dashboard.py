"""Module inspection with Streamlit."""
import logging
import sys

import nnrl.utils as ptu
import numpy as np
import pandas as pd
import ray
import streamlit as st
import torch
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, DataRange1d
from bokeh.plotting import figure
from ray.rllib import SampleBatch

import raylab
import raylab.utils.dictionaries as dutil
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

    y_max = len(columns)
    scatter_plots = []
    for yidx, y_col in enumerate(columns):
        scatter_plots += [[]]
        scatter_plots[yidx] += [None] * yidx
        for xidx in range(yidx, y_max):
            x_col = columns[xidx]

            pic = figure(
                x_range=xdr,
                y_range=ydr,
                plot_width=200,
                plot_height=200,
                min_border_left=2,
                min_border_right=2,
                min_border_top=2,
                min_border_bottom=2,
                tools="",
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

            pic.yaxis.axis_label = ""
            pic.yaxis.visible = False
            pic.xaxis.axis_label = ""
            pic.xaxis.visible = False

            scatter_plots[yidx] += [pic]

    return gridplot(scatter_plots, sizing_mode="scale_both")


def make_histograms(dataset, bins, ranges=()):
    columns = dataset.columns
    limits = tuple(zip(*ranges)) if ranges else (None,) * len(columns)

    xdr = DataRange1d(bounds=None)
    # ydr = DataRange1d(bounds=None)
    # ydr.start = 0

    row = []
    for col, limit in zip(columns, limits):
        hist, edges = np.histogram(dataset[col], density=True, bins=bins, range=limit)
        pic = figure(
            title=col,
            x_range=xdr,
            # y_range=ydr,
            plot_width=200,
            plot_height=240,
            min_border_left=2,
            min_border_right=2,
            min_border_top=2,
            min_border_bottom=42,
            tools="",
        )
        pic.y_range.start = 0
        rend = pic.quad(
            top=hist,
            bottom=0,
            left=edges[:-1],
            right=edges[1:],
            fill_color="navy",
            line_color="white",
            alpha=0.5,
        )
        # pylint:disable=no-member
        xdr.renderers.append(rend)
        # ydr.renderers.append(rend)
        # pylint:enable=no-member

        pic.grid.grid_line_color = "white"
        pic.yaxis.axis_label = ""
        pic.yaxis.visible = False
        row += [pic]

    return gridplot([row], sizing_mode="scale_width")


@st.cache(
    max_entries=1,
    # pylint:disable=protected-access
    hash_funcs={torch.jit.RecursiveScriptModule: hash},
    # pylint:enable=protected-access
)
def get_model_samples(module, row, n_samples):
    obs = ptu.convert_to_tensor(row[SampleBatch.CUR_OBS], "cpu")
    act = ptu.convert_to_tensor(row[SampleBatch.ACTIONS], "cpu")
    new_obs, _ = module.sample(obs, act, (n_samples,))
    return new_obs.detach().numpy()


def plot_model_distributions(policy, row):
    model = policy.module.model
    n_samples = st.number_input("Number of observation samples", min_value=1, step=1)
    samples = get_model_samples(model, row, n_samples)

    data = {f"obs[{i}]": samples[..., i] for i in range(samples.shape[-1])}
    dataset = pd.DataFrame(data)

    new_obs = row[SampleBatch.NEXT_OBS]
    target_data = {f"new_obs[{i}]": new_obs[i][None] for i in range(new_obs.shape[-1])}
    target_dataset = pd.DataFrame(target_data)

    bins = st.number_input(
        "Histogram bins",
        min_value=1,
        max_value=n_samples,
        value=(1 + n_samples) // 2,
        step=1,
    )
    st.bokeh_chart(make_histograms(dataset, bins))
    st.bokeh_chart(scatter_matrix(dataset, target_dataset))


@st.cache(
    max_entries=1,
    # pylint:disable=protected-access
    hash_funcs={
        torch.jit.RecursiveScriptModule: id,
        torch._C._TensorBase: id,
        torch.Tensor: id,
    },
    # pylint:enable=protected-access
)
def get_actor_outputs(module, row, n_samples):
    obs = ptu.convert_to_tensor(row[SampleBatch.CUR_OBS], "cpu")
    with torch.no_grad():
        acts, logp = module.sample(obs, (n_samples,))
        deterministic, _ = module.deterministic(obs)
        deterministic.unsqueeze_(0)
    log_prob = module.log_prob(obs, acts)
    entropy = -log_prob.mean()
    nll_grad = ptu.flat_grad(entropy, module.parameters())
    return {
        "acts": acts,
        "logp": logp,
        "det": deterministic,
        "log_prob": log_prob.detach(),
        "entropy": entropy.detach(),
        "nll_grad": nll_grad,
    }


def plot_action_distributions(outputs, bins, ranges=()):
    acts, det = map(lambda x: x.numpy(), dutil.get_keys(outputs, "acts", "det"))
    data = {f"act[{i}]": acts[..., i] for i in range(acts.shape[-1])}
    dataset = pd.DataFrame(data)

    det_data = {f"det[{i}]": det[..., i] for i in range(det.shape[-1])}
    det_dataset = pd.DataFrame(det_data)

    st.bokeh_chart(make_histograms(dataset, bins, ranges=ranges))
    st.bokeh_chart(scatter_matrix(dataset, det_dataset))


def plot_logp_stats(outputs, bins):
    st.write("Entropy:", outputs["entropy"])
    st.write("grad_norm(nll):", outputs["nll_grad"].norm(p=2))

    dataset = pd.DataFrame({k: outputs[k].numpy() for k in "logp log_prob".split()})
    st.bokeh_chart(make_histograms(dataset, bins))
    dataset = pd.DataFrame({"nll_grad": outputs["nll_grad"].numpy()})
    st.bokeh_chart(make_histograms(dataset, bins))


def viz_actor_distributions(policy, row):
    actor = policy.module.actor
    n_samples = st.number_input("Number of action samples", min_value=1, step=1)

    outputs = get_actor_outputs(actor, row, n_samples)

    bins = st.number_input(
        "Histogram bins",
        min_value=1,
        max_value=n_samples,
        value=(1 + n_samples) // 2,
        step=1,
    )
    plot_action_distributions(
        outputs, bins, ranges=(policy.action_space.low, policy.action_space.high)
    )
    plot_logp_stats(outputs, bins)


def viz_distributions(policy, rollout, timestep):
    row = list(rollout.rows())[timestep]

    components = []
    if "actor" in policy.module:
        components += ["actor"]
    if "model" in policy.module:
        components += ["model"]

    component = st.selectbox("Inspect module:", options=components)
    if component == "actor":
        viz_actor_distributions(policy, row)
    elif component == "model":
        plot_model_distributions(policy, row)


# https://discuss.streamlit.io/t/how-can-i-clear-a-specific-cache-only/1963/6
@st.cache(allow_output_mutation=True)
def get_rollout(agent):
    return [produce_rollout(agent)]


def main():
    setup()
    agent_id, checkpoint = sys.argv[1], sys.argv[2]

    evaluate = st.checkbox("Use evaluation config")
    agent = load_agent(checkpoint, agent_id, evaluate)

    rollout_wrapper = get_rollout(agent)
    if st.button("Sample new episode"):
        rollout_wrapper.clear()
        rollout_wrapper.append(produce_rollout(agent))

    rollout = rollout_wrapper[0]
    timestep = st.slider("Timestep", min_value=0, max_value=rollout.count - 1)

    viz_distributions(agent.get_policy(), rollout, timestep)


if __name__ == "__main__":
    main()
