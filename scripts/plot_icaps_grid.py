import os.path as osp

import numpy as np
import click
import matplotlib.pyplot as plt
import seaborn as sns
from raylab.viskit import core
from raylab.viskit.plot import latexify


def process_algorithm_name(exp_data):
    if "model_loss" in exp_data.params:
        model_loss = "MLE" if exp_data.params["model_loss"] == "mle" else "MAPO"
        exp_data.flat_params["algorithm"] = model_loss
    else:
        exp_data.flat_params["algorithm"] = "DDPG"
    return exp_data


def process_grad_estimator(exp_data):
    if "grad_estimator" in exp_data.params:
        grad_estimator = exp_data.params["grad_estimator"]
        exp_data.flat_params["grad_estimator"] = (
            "SF" if grad_estimator == "score_function" else "PD"
        )
    else:
        exp_data.flat_params["grad_estimator"] = ""
    return exp_data


def plot_navigation_grid(local_dir):
    """Plot experiment comparative grid."""

    path_fmt = osp.join(local_dir, "{}-Navigation-{}/")
    args = dict(
        x="timesteps_total",
        y="evaluation/episode_reward_mean",
        hue="algorithm",
        style="grad_estimator",
        units=None,
        estimator="mean",
        err_style="band",
    )

    def process_params(exp_data):
        exp_data = process_algorithm_name(exp_data)
        exp_data = process_grad_estimator(exp_data)
        return exp_data

    cols = "Walks NonDiagCov LinearModel".split()

    with sns.plotting_context("paper"), latexify(
        columns=2, fig_width=8, fig_height=((np.sqrt(5) - 1.0) / 2.0) * 4
    ):
        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)

        titles = "Random Walks", "Non Diagonal Covariance", "Linear Model"
        for idx, (ax, col, title) in enumerate(zip(axes, cols, titles)):
            exps_data = core.load_exps_data(
                path_fmt.format("MAPO", col)
            ) + core.load_exps_data(path_fmt.format("SOP", col))
            exps_data = list(map(process_params, exps_data))
            if not exps_data:
                continue

            core.insert_params_dataframe(exps_data, "algorithm", "grad_estimator")
            selectors, titles = core.filter_and_split_experiments(exps_data)
            inst = core.lineplot_instructions(selectors, titles, **args)[0]
            plot_kwargs = inst["lineplot_kwargs"]
            plot_kwargs["hue_order"] = ["MAPO", "MLE", "DDPG"]
            sns.lineplot(
                ax=ax,
                legend="full",
                palette=sns.color_palette(
                    ["#1f77b4", "#ff7f0e", "grey"],
                    n_colors=len(plot_kwargs["hue_order"]),
                ),
                **plot_kwargs,
            )
            ax.set_xlabel("Timesteps")
            ax.set_title(title)
            if idx == 0:
                ax.set_ylabel("Episode Reward Mean")
            if idx != 2:
                ax.get_legend().remove()

        # Just some formatting niceness:
        # x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

        fig.tight_layout()
        plt.savefig("navigation.pdf")


def plot_reservoir_grid(local_dir):
    """Plot experiment comparative grid."""

    path_fmt = osp.join(local_dir, "{}-Reservoir-{}/")
    args = dict(
        x="timesteps_total",
        y="evaluation/episode_reward_mean",
        hue="algorithm",
        style="grad_estimator",
        units=None,
        estimator="mean",
        err_style="band",
    )

    def process_params(exp_data):
        exp_data = process_algorithm_name(exp_data)
        exp_data = process_grad_estimator(exp_data)
        return exp_data

    cols = "Walks FullModel LinearModel".split()

    with sns.plotting_context("paper"), latexify(
        columns=2, fig_width=8, fig_height=((np.sqrt(5) - 1.0) / 2.0) * 4
    ):
        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)

        titles = "Random Walks", "Original", "Linear Model"
        for j, (col, title) in enumerate(zip(cols, titles)):
            exps_data = core.load_exps_data(
                path_fmt.format("MAPO", col)
            ) + core.load_exps_data(path_fmt.format("SOP", col))
            exps_data = list(map(process_params, exps_data))
            if not exps_data:
                continue

            core.insert_params_dataframe(exps_data, "algorithm", "grad_estimator")
            selectors, titles = core.filter_and_split_experiments(exps_data)
            plot_inst = core.lineplot_instructions(selectors, titles, **args)[0]
            plot_kwargs = plot_inst["lineplot_kwargs"]
            plot_kwargs["hue_order"] = ["MAPO", "MLE", "DDPG"]
            sns.lineplot(
                ax=axes[j],
                legend="full" if j == 2 else False,
                palette=sns.color_palette(
                    ["#1f77b4", "#ff7f0e", "grey"],
                    n_colors=len(plot_kwargs["hue_order"]),
                ),
                **plot_kwargs,
            )

            axes[j].set_xlabel("Timesteps")
            axes[j].set_title(title)
            if j == 0:
                axes[j].set_ylabel("Episode Reward Mean")

        # Just some formatting niceness:
        # x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

        fig.tight_layout()
        plt.savefig("reservoir.pdf")


def plot_hvac_grid(local_dir):
    """Plot experiment comparative grid."""

    path_fmt = osp.join(local_dir, "{}-HVAC-{}/")
    args = dict(
        # x="timesteps_total",
        x="iterations_since_restore",
        y="evaluation/episode_reward_mean",
        hue="algorithm",
        style="grad_estimator",
        units=None,
        estimator="mean",
        err_style="band",
    )

    def process_params(exp_data):
        exp_data = process_algorithm_name(exp_data)
        exp_data = process_grad_estimator(exp_data)
        return exp_data

    cols = "Walks Original LinearModel".split()

    with sns.plotting_context("paper"):
        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 10))

        for j, col in enumerate(cols):
            exps_data = core.load_exps_data(
                path_fmt.format("MAPO", col)
            ) + core.load_exps_data(path_fmt.format("SOP", col))
            exps_data = list(map(process_params, exps_data))
            if not exps_data:
                continue

            core.insert_params_dataframe(exps_data, "algorithm", "grad_estimator")
            selectors, titles = core.filter_and_split_experiments(exps_data)
            inst = core.lineplot_instructions(selectors, titles, **args)[0]
            plot_kwargs = inst["lineplot_kwargs"]
            sns.lineplot(
                ax=axes[j],
                legend="full" if j == 2 else False,
                palette=sns.color_palette(
                    ["#1f77b4", "#ff7f0e", "grey"],
                    n_colors=len(plot_kwargs["hue_order"]),
                ),
                **plot_kwargs,
            )

        # Just some formatting niceness:
        # x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

        pad = 5  # in points
        for ax, col in zip(axes, cols):
            ax.annotate(
                col,
                xy=(0.5, 1),
                xytext=(0, pad),
                xycoords="axes fraction",
                textcoords="offset points",
                size="large",
                ha="center",
                va="baseline",
            )

        fig.tight_layout()
        plt.show()


@click.command()
@click.option(
    "--local-dir",
    "-l",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    default="data/",
    show_default=True,
    help="",
)
@click.option(
    "--alg-name",
    "-a",
    type=click.Choice(["Navigation", "Reservoir", "HVAC"]),
    default="Navigation",
    show_default=True,
    help="Name of algorithm to plot results from.",
)
def cli(local_dir, alg_name):
    """Plot grid of experiment comparisons."""
    if alg_name == "Navigation":
        plot_navigation_grid(local_dir)
    elif alg_name == "Reservoir":
        plot_reservoir_grid(local_dir)
    elif alg_name == "HVAC":
        plot_hvac_grid(local_dir)
    else:
        raise ValueError(f"Unrecognized algorithm name {alg_name}")


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
