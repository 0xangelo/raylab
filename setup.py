# pylint: disable=missing-docstring
from setuptools import setup, find_packages


def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


setup(
    name="raylab",
    version="0.6.4",
    author="Ângelo G. Lovatto",
    author_email="angelolovatto@gmail.com",
    description="Reinforcement learning algorithms in RLlib and PyTorch.",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/angelolovatto/raylab",
    packages=find_packages(),
    install_requires=[
        "bokeh",
        "Click",
        "ray[rllib,dashboard]>=0.8.5",
        "streamlit",
        "torch",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points="""
        [console_scripts]
        raylab=raylab.cli:raylab
    """,
)
