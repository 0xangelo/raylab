# pylint: disable=missing-docstring
from setuptools import setup

setup(
    name="raylab",
    version="0.5.0",
    py_modules=["raylab"],
    install_requires=[
        "Click",
        "matplotlib",
        "numpy",
        "pandas",
        "seaborn",
        "requests",
        "ray[rllib]==0.8.1",
        "gym",
    ],
    entry_points="""
        [console_scripts]
        raylab=raylab.cli:cli
    """,
)
