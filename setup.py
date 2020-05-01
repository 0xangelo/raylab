# pylint: disable=missing-docstring
from setuptools import setup

setup(
    name="raylab",
    version="0.6.1",
    py_modules=["raylab"],
    install_requires=[
        "Click",
        "matplotlib",
        "pandas",
        "seaborn",
        "requests",
        "ray[rllib]",
    ],
    entry_points="""
        [console_scripts]
        raylab=raylab.cli:cli
    """,
)
