import core.nim as nim
from agents.rand import RandomAgent
from agents.qlearning import QLearningAgent
from agents.dqn import DQNAgent
from agents.alpha_beta import AlphaBetaAgent
from agents.maths_optimal import OptimalAgent
import random

players = [RandomAgent(), QLearningAgent(), DQNAgent(), AlphaBetaAgent(), OptimalAgent()]

ROUNDS = 1000

win_rate = [[0.0 for _ in range(len(players))] for _ in range(len(players))]


for i in range(len(players)):
    for j in range(i + 1, len(players)):
        print(f"{players[i].name()} vs {players[j].name()}")
        win_rate[i][j], win_rate[j][i] = nim.play(players[i], players[j], ROUNDS)

print(win_rate)

table = r"|对方\己方胜率\己方|"
for agent in players:
    table += agent.name() + "|"
table += '\n'
table += "|"

for i in range(len(players)+1):
    table+="---|"
table += '\n'

for i in range(len(players)):
    table += f"|对方是 {players[i].name()}|"
    for j in range(len(players)):
        if i==j:
            table += r" \ |"
        else:
            table += f"{win_rate[j][i]*100:.1f}%|"
    table += '\n'

with open("result.md", "w") as f:
    f.write(table)