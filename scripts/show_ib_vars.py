# pylint:disable=all
"""
Industrial Benchmark inspection
"""
import math
import functools

import streamlit as st
import torch
import numpy as np
import pandas as pd
import bokeh
import bokeh.palettes
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Span
from raylab.cli.viskit import core
from raylab.envs.industrial_benchmark.goldstone.torch.dynamics import TorchDynamics

from bokeh_surface3d import Surface3d


# st.sidebar.header("Sidebar")
# st.slider("slider", min_value=0.0, max_value=10.0)
# st.sidebar.slider
# with st.echo():
#     for i in range(5):
#         print(f"Iteration {i}")


"""
# Plot Industrial Benchmark
"""


def effective_shift_vs_time(dataframe):
    p = figure(title="MyFigure")
    p.line(df["time"], df["he"])
    return p


def default_goldstone_figure(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if "p" not in kwargs:
            p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
            p.xaxis.axis_label = "phi"
            p.yaxis.axis_label = "effective shift"
            kwargs["p"] = p
        return func(*args, **kwargs)

    return wrapped


@default_goldstone_figure
def goldstone_dynamics_landscape(p=None):
    p.title.text = "Mis-Calibration"
    dynamics = TorchDynamics(
        number_steps=24,
        max_required_step=np.sin(15.0 / 180.0 * np.pi),
        safe_zone=np.sin(15.0 / 180.0 * np.pi) / 2,
    )
    phi = torch.linspace(-6, 6, steps=1000)
    effective_shift = torch.linspace(-1.5, 1.5, steps=1000)
    X, Y = torch.meshgrid(phi, effective_shift)
    Z = dynamics.reward(X, Y)

    p.image(
        image=[Z.T.numpy()],
        x=-6,
        y=-1.5,
        dw=12,
        dh=3,
        palette=bokeh.palettes.viridis(256),
    )
    return p


@default_goldstone_figure
def goldstone_rmin_and_ropt(p=None):
    dynamics = TorchDynamics(
        number_steps=24,
        max_required_step=np.sin(15.0 / 180.0 * np.pi),
        safe_zone=np.sin(15.0 / 180.0 * np.pi) / 2,
    )
    phi = torch.linspace(-6, 6, steps=1000)
    rho_s = dynamics._compute_rhos(phi)
    r_min = dynamics._compute_rmin(rho_s)
    r_opt = dynamics._compute_ropt(rho_s)
    p.line(phi.numpy(), r_min.numpy(), legend_label="r_min", color="red")
    p.line(phi.numpy(), r_opt.numpy(), legend_label="r_opt", color="blue")
    return p


@default_goldstone_figure
def goldstone_rhos(p=None):
    dynamics = TorchDynamics(
        number_steps=24,
        max_required_step=np.sin(15.0 / 180.0 * np.pi),
        safe_zone=np.sin(15.0 / 180.0 * np.pi) / 2,
    )
    phi = torch.linspace(-6, 6, steps=1000)
    rho_s = dynamics._compute_rhos(phi)
    p.line(phi.numpy(), rho_s.numpy(), legend_label="rho_s", color="black")
    return p


@default_goldstone_figure
def goldstone_q_threshold(p=None):
    q = -math.sqrt(1 / 27)
    rho_s = 8 * 0.2924 * q / -0.6367
    phi = np.arcsin(rho_s) * 12 / math.pi
    span = Span(
        location=phi,
        dimension="height",
        line_color="gray",
        line_dash="dashed",
        line_width=3,
    )
    p.add_layout(span)
    span = Span(
        location=-phi,
        dimension="height",
        line_color="gray",
        line_dash="dashed",
        line_width=3,
    )
    p.add_layout(span)
    return p


@default_goldstone_figure
def goldstone_q(p=None):
    dynamics = TorchDynamics(
        number_steps=24,
        max_required_step=np.sin(15.0 / 180.0 * np.pi),
        safe_zone=np.sin(15.0 / 180.0 * np.pi) / 2,
    )
    phi = torch.linspace(-6, 6, steps=1000)
    rho_s = dynamics._compute_rhos(phi)
    q = dynamics._compute_q(rho_s)
    p.line(phi.numpy(), q.numpy(), legend_label="q", color="black")
    return p


@default_goldstone_figure
def goldstone_u(p=None):
    dynamics = TorchDynamics(
        number_steps=24,
        max_required_step=np.sin(15.0 / 180.0 * np.pi),
        safe_zone=np.sin(15.0 / 180.0 * np.pi) / 2,
    )
    phi = torch.linspace(-6, 6, steps=1000)
    rho_s = dynamics._compute_rhos(phi)
    q = dynamics._compute_q(rho_s)
    varrho = rho_s.sign()
    u = dynamics._compute_u(q, varrho)
    p.line(phi.numpy(), u.numpy(), legend_label="u", color="orange")
    return p


@default_goldstone_figure
def scatter_effective_shift(dataframe, p=None):
    p.scatter(x=df["gs_phi_idx"], y=df["he"])
    return p


def goldstone_dynamics_3d():
    dynamics = TorchDynamics(
        number_steps=24,
        max_required_step=np.sin(15.0 / 180.0 * np.pi),
        safe_zone=np.sin(15.0 / 180.0 * np.pi) / 2,
    )
    phi = torch.linspace(-6, 6, steps=100)
    effective_shift = torch.linspace(-1.5, 1.5, steps=100)
    X, Y = torch.meshgrid(phi, effective_shift)
    Z = dynamics.reward(X, Y)

    source = ColumnDataSource(
        data=dict(
            phi=X.numpy().ravel(),
            effective_shift=Y.numpy().ravel() * 4,
            z=Z.numpy().ravel(),
        )
    )
    surface = Surface3d(
        x="phi", y="effective_shift", z="z", data_source=source, width=600, height=600
    )
    return surface


# @st.cache
# def get_data():
#     # path = "/Users/angelolovatto/Repositories/personal/raylab/data/MAPO/20200127"
#     path = "/Users/angelolovatto/Repositories/personal/raylab/data/episodes.csv"
#     path = "/Users/angelolovatto/Repositories/personal/raylab/data/SoftAC/20200215/episodes.csv"
#     return pd.read_csv(path)

p = goldstone_dynamics_landscape()
p = goldstone_rhos(p=p)
# p = goldstone_q_threshold(p=p)
# p = goldstone_rmin_and_ropt(p=p)
# p = goldstone_q(p=p)
# p = goldstone_u(p=p)

df = None
path = st.file_uploader("Choose a CSV file", type="csv")
if path is not None:
    df = pd.read_csv(path)
    p = scatter_effective_shift(df, p=p)
    # st.write(df.head())
    # st.bokeh_chart(effective_shift_vs_time(df))
    # st.bokeh_chart(goldstone_dynamics_3d())

st.bokeh_chart(p)
# show(goldstone_dynamics_3d())
