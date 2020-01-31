# pylint:disable=missing-docstring
import click
import seaborn as sns
import matplotlib.pyplot as plt

from raylab.cli.viskit import core
from raylab.cli.viskit.plot import plot_figures


@click.command()
@click.option(
    "--mapo",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True),
    help="",
)
@click.option(
    "--sop",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True),
    help="",
)
@click.option(
    "--sac",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True),
    help="",
)
def main(mapo, sop, sac):
    mapo_data = core.load_exps_data(mapo)
    for exp_data in mapo_data:
        exp_data.flat_params["agent"] = "MAPO"

    sop_data = core.load_exps_data(sop)
    for exp_data in sop_data:
        exp_data.flat_params["agent"] = "SOP"

    sac_data = core.load_exps_data(sac)
    for exp_data in sac_data:
        exp_data.flat_params["agent"] = "SAC"

    exps_data = [
        exp_data
        for exps_data in (mapo_data, sop_data, sac_data)
        for exp_data in exps_data
    ]
    core.insert_params_dataframe(exps_data, "agent")

    selectors, titles = core.filter_and_split_experiments(exps_data)
    instructions = core.lineplot_instructions(
        selectors,
        titles,
        x="timesteps_total",
        y="evaluation/episode_reward_mean",
        hue="agent",
        style="agent",
        legend="full",
    )

    with sns.plotting_context("paper"), sns.axes_style("darkgrid"):
        plot_figures(instructions)
        plt.show()


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
