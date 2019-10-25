"""Registry of algorithm names for `rllib train --run=<alg_name>`"""


def _import_naf():
    from raylab.algorithms.naf.naf import NAFTrainer

    return NAFTrainer


def _import_svg_inf():
    from raylab.algorithms.svg.svg_inf import SVGInfTrainer

    return SVGInfTrainer


def _import_svg_one():
    from raylab.algorithms.svg.svg_one import SVGOneTrainer

    return SVGOneTrainer


def _import_sac():
    from raylab.algorithms.sac.sac import SACTrainer

    return SACTrainer


def _import_sop():
    from raylab.algorithms.sop.sop import SOPTrainer

    return SOPTrainer


ALGORITHMS = {
    "NAF": _import_naf,
    "SVG(inf)": _import_svg_inf,
    "SVG(1)": _import_svg_one,
    "SAC": _import_sac,
    "SOP": _import_sop,
}
