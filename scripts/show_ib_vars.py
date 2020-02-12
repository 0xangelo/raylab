"""
# Plot Industrial Benchmark
"""
import streamlit as st
import numpy as np
import pandas as pd
import bokeh
from bokeh.plotting import figure

from raylab.cli.viskit import core


st.sidebar.header("Sidebar")
# st.slider("slider", min_value=0.0, max_value=10.0)

# st.sidebar.slider

# with st.echo():
#     for i in range(5):
#         print(f"Iteration {i}")


@st.cache
def get_data():
    # path = "/Users/angelolovatto/Repositories/personal/raylab/data/MAPO/20200127"
    path = "/Users/angelolovatto/Repositories/personal/raylab/data/episodes.csv"
    return pd.read_csv(path)


df = get_data()

df

p = figure(title="MyFigure")

p.line(df["time"], df["he"])


st.bokeh_chart(p)

p = figure(title="miscalibration")
p.xaxis.axis_label = "$\phi$"
p.yaxis.axis_label = "$h^e$"
p.scatter(x=df["gs_phi_idx"], y=df["he"])
st.bokeh_chart(p)
