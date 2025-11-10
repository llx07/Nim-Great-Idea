import math
import os
import sys

# Allow running this file separately
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import core.nim as nim
import random
import copy

import functools 

class AlphaBetaAgent(nim.Agent):
    @staticmethod
    def _get_available_actions(piles) -> list[tuple[int, int]]:
        """Return all available actions according to `piles`.

        Returns:
            A list of tuple in format (pile_number, count).
        """
        actions = []
        for pile_number, stones in enumerate(piles):
            for count in range(1, stones + 1):
                actions.append((pile_number, count))
        return actions


    @functools.cache
    def alpha_beta(
        self, piles: tuple[int], alpha: float, beta: float, player: int
    ) -> float:
        """Run alpha beta search. Return the score of a state.

        player = 0: self (maximizing node)
        player = 1: other (minimizing node)
        """

        if all(x == 0 for x in piles):
            return -1 if player == 0 else 1

        actions = AlphaBetaAgent._get_available_actions(piles)

        if player == 0:
            max_score = float("-inf")
            for move in actions:
                idx, cnt = move
                new_piles = tuple(x if i!=idx else x-cnt for i,x in enumerate(piles))
                score = self.alpha_beta(new_piles, alpha, beta, 1)
                max_score = max(max_score, score)
                alpha = max(alpha, max_score)
                if alpha >= beta:
                    break
            # print(f"{piles=}, {max_score=}")
            assert not math.isinf(max_score)
            return max_score
        else:
            min_score = float("inf")
            for move in actions:
                idx, cnt = move
                new_piles = tuple(x if i!=idx else x-cnt for i,x in enumerate(piles))
                score = self.alpha_beta(new_piles, alpha, beta, 0)
                min_score = min(min_score, score)
                beta = min(beta, min_score)
                if beta <= alpha:
                    break
            assert not math.isinf(min_score)
            return min_score

    def choose_action(self, state: nim.NimEnv) -> tuple[int, int]:
        """Get the best action"""
        best_move = None
        best_score = float("-inf")
        # print(f"{state.piles=}")

        piles = state.piles.copy()
        for move in state.get_available_actions():
            idx, cnt = move
            piles[idx] -= cnt
            move_score = self.alpha_beta(tuple(piles), float("-inf"), float("inf"), 1)
            piles[idx] += cnt

            if best_move is None or move_score > best_score:
                best_score = move_score
                best_move = move

        print(f"{state.piles=} {best_move=}")
        return best_move  # type: ignore (best_move will never be None)

    def name(self) -> str:
        return "Alpha-Beta 剪枝"
