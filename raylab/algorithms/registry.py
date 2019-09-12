"""Registry of algorithm names for `rllib train --run=<alg_name>`"""


def _import_naf():
    from raylab.algorithms.naf.naf import NAFTrainer

    return NAFTrainer


ALGORITHMS = {"NAF": _import_naf}
