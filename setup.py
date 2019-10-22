# pylint: disable=missing-docstring
from setuptools import setup

setup(
    name="raylab",
    version="0.2.0",
    py_modules=["raylab"],
    install_requires=[
        "Click",
        "matplotlib",
        "numpy",
        "pandas",
        "seaborn",
        "ray",
        "gym",
    ],
    entry_points="""
        [console_scripts]
        raylab=raylab.cli:cli
        viskit=raylab.viskit.plot:cli
    """,
)
