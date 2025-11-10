# 策略：
# 1. 计算所有堆石子数的异或和（尼姆和）
# 2. 当尼姆和为0时，当前玩家处于必败态，无论怎么操作都会让对手进入必胜态
# 3. 当尼姆和不为0时，当前玩家可以通过一次操作使尼姆和变为0，让对手进入必败态

import os
import sys

# Allow running this file separately
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import core.nim as nim


class OptimalAgent(nim.Agent):
    """Optimal agent for Nim game using Nim-sum strategy"""
    
    def _nim_sum(self, piles: list[int]) -> int:
        """Calculate the XOR sum (Nim-sum) of all pile sizes"""
        xor_sum = 0
        for stones in piles:
            xor_sum ^= stones
        return xor_sum
    
    def choose_action(self, state: nim.NimEnv) -> tuple[int, int]:
        """Choose optimal action based on Nim-sum strategy"""
        piles = state.piles
        xor_sum = self._nim_sum(piles)
        
        # If Nim-sum is not zero, find the move that makes it zero
        if xor_sum != 0:
            for pile_idx, stones in enumerate(piles):
                # Check if this pile can be used to balance the Nim-sum
                if (stones ^ xor_sum) < stones:
                    remove_count = stones - (stones ^ xor_sum)
                    return (pile_idx, remove_count)
        
        # If Nim-sum is zero (losing position), return first available action
        return state.get_available_actions()[0]
    
    def name(self) -> str:
        return "数学最优"