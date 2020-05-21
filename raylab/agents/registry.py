"""Registry of agent names for `rllib train --run=<alg_name>`"""


def _import_naf():
    from raylab.agents.naf.naf import NAFTrainer

    return NAFTrainer


def _import_svg_inf():
    from raylab.agents.svg import SVGInfTrainer

    return SVGInfTrainer


def _import_svg_one():
    from raylab.agents.svg import SVGOneTrainer

    return SVGOneTrainer


def _import_sac():
    from raylab.agents.sac.sac import SACTrainer

    return SACTrainer


def _import_sop():
    from raylab.agents.sop.sop import SOPTrainer

    return SOPTrainer


def _import_mapo():
    from raylab.agents.mapo import MAPOTrainer

    return MAPOTrainer


def _import_trpo():
    from raylab.agents.trpo.trpo import TRPOTrainer

    return TRPOTrainer


def _import_soft_svg():
    from raylab.agents.svg import SoftSVGTrainer

    return SoftSVGTrainer


def _import_acktr():
    from raylab.agents.acktr.acktr import ACKTRTrainer

    return ACKTRTrainer


AGENTS = {
    "NAF": _import_naf,
    "SVG(inf)": _import_svg_inf,
    "SVG(1)": _import_svg_one,
    "SoftAC": _import_sac,
    "SOP": _import_sop,
    "MAPO": _import_mapo,
    "TRPO": _import_trpo,
    "SoftSVG": _import_soft_svg,
    "ACKTR": _import_acktr,
}
