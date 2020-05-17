# pylint: disable=missing-docstring
import functools
from math import sqrt
from contextlib import nullcontext

import click

from ..utils import initialize_raylab


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
    import matplotlib

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
    import matplotlib.pyplot as plt
    import seaborn as sns

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


def decorate_lineplot_kwargs(func):
    @click.option(
        "--logdir",
        "-p",
        multiple=True,
        help="directories to look for progress and parameter data.",
    )
    @click.option(
        "--split",
        default=None,
        show_default=True,
        help="plot one figure for each set of trials "
        "grouped by this key in parameters.",
    )
    @click.option(
        "--include",
        "-i",
        type=(str, str),
        multiple=True,
        help="only include trials with this key-value pair in parameters.",
    )
    @click.option(
        "--exclude",
        "-e",
        type=(str, str),
        multiple=True,
        help="exclude trials with this key-value pair in parameters.",
    )
    @click.option(
        "--subskey",
        "-k",
        type=(str, str),
        multiple=True,
        help="for this key-value pair, substitute 'value' for 'key' in parameters.",
    )
    @click.option(
        "--subsval",
        "-v",
        type=(str, str),
        multiple=True,
        help="for this key-value pair, substitute 'value' "
        "for all 'key's in parameters.",
    )
    @click.option(
        "--progress", default="progress", help="prefix for experiment progress files."
    )
    @click.option(
        "--params", default="params", help="prefix for experiment parameters files."
    )
    @click.option(
        "--xaxis",
        "-x",
        "x",
        default="timesteps_total",
        show_default=True,
        help="names of variables in ``data``, optional. "
        "Input data variables; must be numeric. "
        "Should pass reference columns in ``data``.",
    )
    @click.option(
        "--yaxis",
        "-y",
        "y",
        default="episode_reward_mean",
        show_default=True,
        help="names of variables in ``data``, optional. "
        "Input data variables; must be numeric. "
        "Should pass reference columns in ``data``.",
    )
    @click.option(
        "--hue",
        default=None,
        show_default=True,
        help="name of variables in ``data`` or vector data, optional. "
        "Grouping variable that will produce lines with different colors. "
        "Can be either categorical or numeric, although color mapping will "
        "behave differently in latter case.",
    )
    @click.option(
        "--size",
        default=None,
        show_default=True,
        help="name of variables in ``data`` or vector data, optional. "
        "Grouping variable that will produce lines with different widths. "
        "Can be either categorical or numeric, although size mapping will "
        "behave differently in latter case.",
    )
    @click.option(
        "--style",
        default=None,
        show_default=True,
        help="name of variables in ``data`` or vector data, optional. "
        "Grouping variable that will produce lines with different dashes "
        "and/or markers. Can have a numeric dtype but will always be treated "
        "as categorical.",
    )
    @click.option(
        "--palette",
        default="deep",
        show_default=True,
        help="palette name, optional. "
        "Colors to use for the different levels of the ``hue`` variable. Should "
        "be something that can be interpreted by :func:`color_palette`.",
    )
    @click.option(
        "--units",
        default=None,
        show_default=True,
        help="Grouping variable identifying sampling units. When used, a separate "
        "line will be drawn for each unit with appropriate semantics, but no "
        "legend entry will be added. Useful for showing distribution of "
        "experimental replicates when exact identities are not needed.",
    )
    @click.option(
        "--estimator",
        default="mean",
        show_default=True,
        help="name of pandas method or None, optional. "
        "Method for aggregating across multiple observations of the ``y`` "
        "variable at the same ``x`` level. If ``None``, all observations will "
        "be drawn.",
    )
    @click.option(
        "--err-style",
        default="band",
        show_default=True,
        help="'band' or 'bars', optional. "
        "Whether to draw the confidence intervals with translucent error bands "
        "or discrete error bars.",
    )
    @click.option(
        "--legend",
        "-l",
        type=click.Choice(["full", "brief"]),
        default="brief",
        show_default=True,
        help="'brief', 'full', or False, optional "
        "How to draw the legend. If 'brief', numeric ``hue`` and ``size`` "
        "variables will be represented with a sample of evenly spaced values. "
        "If 'full', every group will get an entry in the legend. If ``False``, "
        "no legend data is added and no legend is drawn.",
    )
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapped


def load_and_create_instructions(**args):
    """CLI to draw Seaborn's lineplot from experiment data."""
    import raylab.utils.exp_data as exp_util
    from . import core

    exps_data = exp_util.load_exps_data(
        args["logdir"], progress_prefix=args["progress"], config_prefix=args["params"]
    )
    core.rename_params(exps_data, args["subskey"], args["subsval"])
    core.insert_params_dataframe(exps_data, args["hue"], args["size"], args["style"])

    selectors, titles = exp_util.filter_and_split_experiments(
        exps_data, split=args["split"], include=args["include"], exclude=args["exclude"]
    )

    return core.lineplot_instructions(
        selectors,
        titles,
        x=args["x"],
        y=args["y"],
        hue=args["hue"],
        size=args["size"],
        style=args["style"],
        units=args["units"],
        estimator=args["estimator"] if args["units"] is None else None,
        err_style=args["err_style"],
        legend=args["legend"],
    )


@click.command()
@decorate_lineplot_kwargs
@click.option(
    "--context",
    type=click.Choice("paper notebook talk poster".split()),
    default="paper",
    show_default=True,
    help="This affects things like the size of the labels, lines, and other "
    "elements of the plot, but not the overall style. The base context "
    "is 'notebook', and the other contexts are 'paper', 'talk', and 'poster', "
    "which are version of the notebook parameters scaled by .8, 1.3, and 1.6, "
    "respectively.",
)
@click.option(
    "--axstyle",
    type=click.Choice("darkgrid whitegrid dark white ticks".split()),
    default="darkgrid",
    show_default=True,
    help="This affects things like the color of the axes, whether a grid is "
    "enabled by default, and other aesthetic elements.",
)
@initialize_raylab
def plot(**args):
    """Draw lineplots of the relevant variables and display them on screen."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_instructions = load_and_create_instructions(**args)

    plotting_context = sns.plotting_context(args["context"])
    axes_style = sns.axes_style(args["axstyle"])
    with plotting_context, axes_style:
        plot_figures(plot_instructions)
        plt.show()


@click.command()
@decorate_lineplot_kwargs
@click.option(
    "--latex/--no-latex",
    default=False,
    help="whether to save the plot in a LaTex friendly format.",
)
@click.option(
    "--latexcol",
    default=2,
    show_default=True,
    help="resize the image to fit the desired number of columns.",
)
@click.option(
    "--facecolor",
    "-fc",
    default="#F5F5F5",
    show_default=True,
    help="http://latexcolor.com",
)
@click.option("--out", "-o", required=True, help="file to save the output image to.")
@initialize_raylab
def plot_export(**args):
    """Draw lineplots of the relevant variables and save them as files."""
    plot_instructions = load_and_create_instructions(**args)
    import matplotlib.pyplot as plt

    if args["latex"]:
        import matplotlib

        matplotlib.rcParams.update(
            {
                "backend": "ps",
                "text.latex.preamble": ["\\usepackage{gensymb}"],
                "text.usetex": True,
            }
        )
        for plot_inst in plot_instructions:
            lineplot_kwargs = plot_inst["lineplot_kwargs"]
            lineplot_kwargs["data"].rename = lineplot_kwargs["data"].rename(
                columns=lambda s: s.replace("_", " ").capitalize(), inplace=True
            )

    latex_style = latexify(columns=args["latexcol"]) if args["latex"] else nullcontext()
    with latex_style:
        plot_figures(plot_instructions)
        plt.savefig(args["out"], facecolor=args["facecolor"])
