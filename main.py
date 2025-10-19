import nim
from agent_random import AgentRandom
from agent_qlearning import QLearningAgent
import random

players = [AgentRandom(), QLearningAgent()]

rounds = 1000
win_count = [0, 0]
for _ in range(rounds):
    game = nim.NimEnv([random.randint(5, 9) for _ in range(4)])
    while game.winner is None:
        game.move(players[game.player].choose_action(game))
    win_count[game.winner] += 1


print(f"player1: win rate = {win_count[0]/ rounds}")
print(f"player2: win rate = {win_count[1]/ rounds}")
