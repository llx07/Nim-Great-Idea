import os
import sys

# Allow running this file separately
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import core.nim as nim
import pickle
import random
import copy


class AlphaBetaAgent(nim.Agent):
    def alpha_beta(self):
        pass

    def name(self) -> str:
        return "Alpha-Beta 剪枝"