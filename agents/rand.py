import core.nim as nim
import random

class RandomAgent(nim.Agent):
    def choose_action(self, state: nim.NimEnv) -> tuple[int, int]:
        return random.choice(state.get_available_actions())