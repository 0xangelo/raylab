#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    README = readme_file.read()

with open("HISTORY.rst") as history_file:
    HISTORY = history_file.read()

REQUIREMENTS = [
    "Click>=7.0",
    "bokeh",
    "ray[rllib,dashboard]>=0.8.5",
    "streamlit",
    "torch",
]

SETUP_REQUIREMENTS = [
    "pytest-runner",
]

TEST_REQUIREMENTS = [
    "pytest>=3",
]

setup(
    author="Ângelo Gregório Lovatto",
    author_email="angelolovatto@gmail.com",
    python_requires="~=3.7.0",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    description="Reinforcement learning algorithms in RLlib and PyTorch.",
    entry_points="""
        [console_scripts]
        raylab=raylab.cli:raylab
    """,
    install_requires=REQUIREMENTS,
    license="MIT license",
    long_description=README + "\n\n" + HISTORY,
    include_package_data=True,
    keywords="raylab",
    name="raylab",
    packages=find_packages(include=["raylab", "raylab.*"]),
    setup_requires=SETUP_REQUIREMENTS,
    test_suite="tests",
    tests_require=TEST_REQUIREMENTS,
    url="https://github.com/angelolovatto/raylab",
    version="0.6.5",
    zip_safe=False,
)
