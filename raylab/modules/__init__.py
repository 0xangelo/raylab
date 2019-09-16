"""Collection of neural network modules in PyTorch."""
import torch.nn as nn

from raylab.modules.fully_connected import FullyConnectedModule
from raylab.modules.state_action_encoder import StateActionEncodingModule
from raylab.modules.tril_matrix import TrilMatrixModule
from raylab.modules.action import ActionModule
from raylab.modules.value import ValueModule
