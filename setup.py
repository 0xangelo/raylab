from setuptools import setup

setup(
    name="raylab",
    version="0.1",
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
