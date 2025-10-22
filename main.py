import core.nim as nim
from agents.rand import RandomAgent
from agents.qlearning import QLearningAgent
from agents.dqn import DQNAgent
import random

players = [RandomAgent(), DQNAgent()]

rounds = 1000
win_count = [0, 0]
for _ in range(rounds):
    game = nim.NimEnv([random.randint(5, 9) for _ in range(4)])
    while game.winner is None:
        # print("player now = ", game.player)
        action = players[game.player].choose_action(game)
        # print(game.piles,game.player, action)
        assert game.move(action) != -1
    win_count[game.winner] += 1


print(f"player1: win rate = {win_count[0]/ rounds}")
print(f"player2: win rate = {win_count[1]/ rounds}")
