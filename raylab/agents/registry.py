"""Registry of agents as trainables for Tune."""
from ray.tune.registry import get_trainable_cls


def _import_naf():
    from raylab.agents.naf import NAFTrainer

    return NAFTrainer


def _import_svg_inf():
    from raylab.agents.svg.inf import SVGInfTrainer

    return SVGInfTrainer


def _import_svg_one():
    from raylab.agents.svg.one import SVGOneTrainer

    return SVGOneTrainer


def _import_sac():
    from raylab.agents.sac import SACTrainer

    return SACTrainer


def _import_sop():
    from raylab.agents.sop import SOPTrainer

    return SOPTrainer


def _import_mapo():
    from raylab.agents.mapo import MAPOTrainer

    return MAPOTrainer


def _import_trpo():
    from raylab.agents.trpo import TRPOTrainer

    return TRPOTrainer


def _import_soft_svg():
    from raylab.agents.svg.soft import SoftSVGTrainer

    return SoftSVGTrainer


def _import_acktr():
    from raylab.agents.acktr import ACKTRTrainer

    return ACKTRTrainer


def _import_mbpo():
    from raylab.agents.mbpo import MBPOTrainer

    return MBPOTrainer


def _import_mage():
    from raylab.agents.mage import MAGETrainer

    return MAGETrainer


def _import_dyna_sac():
    from raylab.agents.sac.dyna import DynaSACTrainer

    return DynaSACTrainer


def _import_mapo_mle():
    from raylab.agents.mapo.mle import MlMAPOTrainer

    return MlMAPOTrainer


def _import_dapo():
    from raylab.agents.mapo.dapo import DAPOTrainer

    return DAPOTrainer


def _import_mapo_plus():
    from raylab.agents.mapo.plus import MAPOPlusTrainer

    return MAPOPlusTrainer


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
    "MBPO": _import_mbpo,
    "MAGE": _import_mage,
    "Dyna-SAC": _import_dyna_sac,
    "MAPO-MLE": _import_mapo_mle,
    "DAPO": _import_dapo,
    "MAPO++": _import_mapo_plus,
}


def get_agent_cls(agent_name):
    """Retrieve agent class from global registry.

    The user must have called `raylab.register_all_agents()` beforehand to
    have access to Raylab's agents.
    """
    return get_trainable_cls(agent_name)
