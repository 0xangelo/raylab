import os.path as osp

import click
import matplotlib.pyplot as plt
import seaborn as sns
from raylab.viskit import core


def process_deceleration_zones(exp_data):
    exp_data.flat_params["num_zones"] = len(
        exp_data.flat_params["env_config/deceleration_zones/decay"]
    )
    return exp_data


def process_algorithm_name(exp_data):
    if "model_loss" in exp_data.params:
        model_loss = "MLE" if exp_data.params["model_loss"] == "mle" else "MAPO"
        grad_estimator = (
            "SF" if exp_data.params["grad_estimator"] == "score_function" else "PD"
        )
        exp_data.flat_params["algorithm"] = model_loss + " " + grad_estimator
    else:
        exp_data.flat_params["algorithm"] = "SOP"
    return exp_data


def plot_navigation_grid(local_dir):
    """Plot experiment comparative grid."""

    path_fmt = osp.join(local_dir, "{}-Navigation-{}/")
    args = dict(
        x="timesteps_total",
        y="evaluation/episode_reward_mean",
        hue="algorithm",
        units=None,
        estimator="mean",
        err_style="band",
    )

    cols = "Walks NonDiagCov LinearModel".split()
    rows = ["{} Zones".format(col) for col in (1, 2, 4)]

    with sns.plotting_context("paper"):
        fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(8, 8))

        for j, col in enumerate(cols):
            exps_data = core.load_exps_data(
                path_fmt.format("MAPO", col)
            ) + core.load_exps_data(path_fmt.format("SOP", col))
            exps_data = list(
                map(process_deceleration_zones, map(process_algorithm_name, exps_data))
            )
            core.insert_params_dataframe(exps_data, "algorithm")
            selectors, titles = core.filter_and_split_experiments(
                exps_data, split="num_zones"
            )
            plot_insts = core.lineplot_instructions(selectors, titles, **args)
            for i, inst in enumerate(plot_insts):
                plot_kwargs = inst["lineplot_kwargs"]
                sns.lineplot(
                    ax=axes[i][j],
                    legend="full" if i == 2 and j == 2 else False,
                    **plot_kwargs,
                )

        # Just some formatting niceness:
        # x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

        pad = 5  # in points
        for ax, col in zip(axes[0], cols):
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
        for ax, row in zip(axes[:, 0], rows):
            ax.annotate(
                row,
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                size="large",
                ha="right",
                va="center",
            )

        fig.tight_layout()
        plt.show()


def plot_reservoir_grid(local_dir):
    """Plot experiment comparative grid."""

    path_fmt = osp.join(local_dir, "{}-Reservoir-{}/")
    args = dict(
        x="timesteps_total",
        y="evaluation/episode_reward_mean",
        hue="algorithm",
        units=None,
        estimator="mean",
        err_style="band",
    )

    cols = "Walks NonDiagCov LinearModel".split()

    with sns.plotting_context("paper"):
        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(4, 8))

        for j, col in enumerate(cols):
            exps_data = core.load_exps_data(
                path_fmt.format("MAPO", col)
            ) + core.load_exps_data(path_fmt.format("SOP", col))
            exps_data = list(
                map(process_deceleration_zones, map(process_algorithm_name, exps_data))
            )
            core.insert_params_dataframe(exps_data, "algorithm")
            selectors, titles = core.filter_and_split_experiments(exps_data)
            plot_inst = core.lineplot_instructions(selectors, titles, **args)[0]
            plot_kwargs = plot_inst["lineplot_kwargs"]
            sns.lineplot(
                ax=axes[1][j], legend="full" if j == 2 else False, **plot_kwargs
            )

        # Just some formatting niceness:
        # x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

        pad = 5  # in points
        for ax, col in zip(axes[0], cols):
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
    type=click.Choice(["Navigation", "Reservoir"]),
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
    else:
        raise ValueError(f"Unrecognized algorithm name {alg_name}")


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
