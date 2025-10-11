import nim
from agent_random import AgentRandom

game = nim.NimEnv()
players = [AgentRandom(), AgentRandom()]

while game.winner is None:
    game.move(players[game.player].choose_action(game))

print(f"Game end with {game.winner=}")