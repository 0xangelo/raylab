from math import sqrt
from contextlib import nullcontext

import click
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from raylab.viskit import core


def latexify(fig_width=None, fig_height=None, columns=1, max_height_inches=8.0):
    """Return matplotlib's RC params for LaTeX plotting.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert columns in [1, 2]

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    if fig_height > max_height_inches:
        print(
            "WARNING: fig_height too large:"
            + fig_height
            + "so will reduce to"
            + max_height_inches
            + "inches."
        )
        fig_height = max_height_inches

    new_params = {
        "axes.labelsize": 8,  # fontsize for x and y labels (was 10)
        "axes.titlesize": 8,
        "legend.fontsize": 8,  # was 10
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": [fig_width, fig_height],
        "font.family": ["serif"],
    }
    return matplotlib.rc_context(rc=new_params)


def plot_figures(plot_instructions):
    for idx, plot_inst in enumerate(plot_instructions):
        print("Dataframe columns for figure {}:".format(idx))
        print(*plot_inst["lineplot_kwargs"]["data"].columns)
        print("Dataframe head for figure {}:".format(idx))
        print(plot_inst["lineplot_kwargs"]["data"].head())

        plt.figure()
        plt.title(plot_inst["title"])
        lineplot_kwargs = plot_inst["lineplot_kwargs"]
        sns.lineplot(**lineplot_kwargs)

        if lineplot_kwargs["data"][lineplot_kwargs["x"]].max() > 5e3:
            # Just some formatting niceness:
            # x-axis scale in scientific notation if max x is large
            plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        plt.tight_layout(pad=0.5)


@click.group()
@click.option("--logdir", "-p", multiple=True)
@click.option("--xaxis", "-x", "x", default="timesteps_total", show_default=True)
@click.option("--yaxis", "-y", "y", default="episode_reward_mean", show_default=True)
@click.option("--hue", default=None)
@click.option("--size", default=None)
@click.option("--style", default=None)
@click.option("--estimator", default="mean", show_default=True)
@click.option("--units", is_flag=True)
@click.option("--split", default=None)
@click.option("--include", "-i", type=(str, str), multiple=True)
@click.option("--exclude", "-e", type=(str, str), multiple=True)
@click.option("--subskey", "-k", type=(str, str), multiple=True)
@click.option("--subsval", "-v", type=(str, str), multiple=True)
@click.option(
    "--legend",
    "-l",
    type=click.Choice(["full", "brief"]),
    default="brief",
    show_default=True,
)
@click.option(
    "--progress", default="progress", help="Prefix for experiment progress files"
)
@click.option(
    "--params", default="params", help="Prefix for experiment parameters files"
)
@click.pass_context
def cli(ctx, **args):
    ctx.obj = args
    exps_data = core.load_exps_data(
        args["logdir"], progress_prefix=args["progress"], config_prefix=args["params"]
    )
    core.rename_params(exps_data, args["subskey"], args["subsval"])
    core.insert_params_dataframe(exps_data, args["hue"], args["size"], args["style"])

    ctx.obj["plot_instructions"] = core.lineplot_instructions(
        exps_data,
        split=args["split"],
        include=args["include"],
        exclude=args["exclude"],
        x=args["x"],
        y=args["y"],
        hue=args["hue"],
        size=args["size"],
        style=args["style"],
        estimator=None if args["units"] else args["estimator"],
        units="unit" if args["units"] else None,
        legend=args["legend"],
        palette=sns.color_palette("deep", 1),
    )


@cli.command()
@click.option("--context", default="paper", show_default=True)
@click.option("--axstyle", default="darkgrid", show_default=True)
@click.pass_context
def show(ctx, **args):
    args.update(ctx.obj)
    plot_instructions = args["plot_instructions"]

    plotting_context = sns.plotting_context(args["context"])
    axes_style = sns.axes_style(args["axstyle"])
    with plotting_context, axes_style:
        plot_figures(plot_instructions)
        plt.show()


@cli.command()
@click.option("--latex/--no-latex", default=False)
@click.option("--latexcol", default=2, show_default=True)
@click.option(
    "--facecolor",
    "-fc",
    default="#F5F5F5",
    show_default=True,
    help="http://latexcolor.com",
)
@click.option("--out", "-o", required=True)
@click.pass_context
def save(ctx, **args):
    args.update(ctx.obj)
    plot_instructions = args["plot_instructions"]
    if args["latex"]:
        matplotlib.rcParams.update(
            {
                "backend": "ps",
                "text.latex.preamble": ["\\usepackage{gensymb}"],
                "text.usetex": True,
            }
        )
        for plot_inst in plot_instructions:
            lineplot_kwargs = plot_inst["lineplot_kwargs"]
            lineplot_kwargs["data"] = lineplot_kwargs["data"].rename(
                columns=lambda s: s.replace("_", "-"), inplace=True
            )

    latex_style = latexify(columns=args["latexcol"]) if args["latex"] else nullcontext()
    with latex_style:
        plot_figures(plot_instructions)
        plt.savefig(args["out"], facecolor=args["facecolor"])
