# pylint: disable=missing-docstring
from setuptools import setup

setup(
    name="raylab",
    version="0.4.0",
    py_modules=["raylab"],
    install_requires=[
        "Click",
        "matplotlib",
        "numpy",
        "pandas",
        "seaborn",
        "requests",
        "ray[rllib]",
        "gym",
    ],
    entry_points="""
        [console_scripts]
        raylab=raylab.cli:cli
    """,
)
