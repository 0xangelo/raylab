"""Collection of neural network modules in PyTorch."""
import torch.nn as nn

from raylab.modules.fully_connected import FullyConnected
from raylab.modules.state_action_encoder import StateActionEncoder
from raylab.modules.tril_matrix import TrilMatrix
from raylab.modules.action_output import ActionOutput
from raylab.modules.value_function import ValueFunction
from raylab.modules.diag_multivariate_normal_params import DiagMultivariateNormalParams
from raylab.modules.lambd import Lambda
