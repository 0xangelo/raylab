# pylint:disable=missing-module-docstring
import functools

import numpy as np
import bokeh
from bokeh.plotting import figure
from bokeh.models import HoverTool


def initialize_raylab(func):
    """Wrap cli to register raylab's algorithms and environments."""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        import raylab

        raylab.register_all_agents()
        raylab.register_all_environments()

        return func(*args, **kwargs)

    return wrapped


def time_series(x_key, y_key, groups, labels, individual=False, standard_error=False):
    """Plot time series with error bands per group."""
    # pylint:disable=too-many-locals,too-many-function-args,too-many-arguments
    pic = figure(title="Plot")
    pic.xaxis.axis_label = x_key
    pic.yaxis.axis_label = y_key
    pic.add_tools(HoverTool(tooltips=[("y", "@y"), ("x", "@x{a}")]))
    palette = bokeh.palettes.cividis(len(labels))

    for idx, (label, group) in enumerate(zip(labels, groups)):
        data = group.extract()
        progresses = [d.progress for d in data]
        # Filter NaN values from plots
        masks = [~np.isnan(p[y_key]) for p in progresses]
        xs_ = [p[x_key][m] for m, p in zip(masks, progresses)]
        ys_ = [p[y_key][m] for m, p in zip(masks, progresses)]
        x_all = np.unique(np.sort(np.concatenate(xs_)))
        all_ys = [
            np.interp(x_all, x, y, left=np.nan, right=np.nan) for x, y in zip(xs_, ys_)
        ]

        if individual:
            for datum, y_i in zip(data, all_ys):
                legend_label = label + "-" + str(datum.params["id"])
                pic.line(x_all, y_i, legend_label=legend_label, color=palette[idx])
        else:
            y_mean = np.nanmean(all_ys, axis=0)
            pic.line(x_all, y_mean, legend_label=label, color=palette[idx])
            dispersion = np.nanstd(all_ys, axis=0)
            if standard_error:
                dispersion /= np.sqrt(np.sum(1 - np.isnan(y_mean), axis=0))
            pic.varea(
                x_all,
                y1=y_mean - dispersion,
                y2=y_mean + dispersion,
                fill_alpha=0.25,
                legend_label=label,
                color=palette[idx],
            )

        pic.legend.location = "bottom_left"
        pic.legend.click_policy = "hide"
    return pic
