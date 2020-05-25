"""Utilities for dynamically importing modules."""
import os.path as osp
import pathlib
import sys
from importlib import import_module


def import_module_from_path(path):
    """Import a module given its path in the filesystem."""
    sys.path.append(osp.dirname(path))
    return import_module(pathlib.Path(path).stem)
