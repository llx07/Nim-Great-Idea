import core.nim as nim
import pickle
import os
import random
import copy


class QLearningAgent(nim.Agent):
    def __init__(self, learning_rate=0.5, discount_factor=0.1) -> None:
        """Initialize the agent. Load q-table from file (if exist)
        Also set alpha and gamma.
        """

        if not os.path.exists("model/qtable.data"):
            self.q = dict()
        else:
            with open("model/qtable.data", "rb") as file:
                self.q = pickle.load(file)
        self.alpha = learning_rate
        self.gamma = discount_factor

    def save(self) -> None:
        """Save the q-table into file."""

        with open("qtable.data", "wb") as file:
            pickle.dump(self.q, file)

    def update(
        self,
        state_before_: list[int],
        action: tuple[int, int],
        state_after_: list[int],
        reward: float,
    ):
        """Update the Q value for the state"""

        state_before = tuple(state_before_)
        state_after = tuple(state_after_)

        if state_before not in self.q:
            self.q[state_before] = dict()
        if action not in self.q[state_before]:
            self.q[state_before][action] = 0

        self.q[state_before][action] = (1 - self.alpha) * self.q[state_before][
            action
        ] + self.alpha * (
            reward + self.gamma * max(self.q.get(state_after, {0: 0}).values())
        )

    def choose_action(self, state: nim.NimEnv) -> tuple[int, int]:
        """Given a state `state`, return an action."""

        if tuple(state.piles) not in self.q:
            return random.choice(state.get_available_actions())
        max_value = None
        max_action = None
        for action, value in self.q[tuple(state.piles)].items():
            if (max_value is None) or (max_value < value):
                max_value = value
                max_action = action
        return max_action  # type: ignore

    def choose_action_with_epsilion(
        self, state: nim.NimEnv, epsilion: float = 0.1
    ) -> tuple[int, int]:
        if random.random() < epsilion:
            return random.choice(state.get_available_actions())
        return self.choose_action(state)


if __name__ == "__main__":
    agent = QLearningAgent()
    for _ in range(2000000):
        if _ % 1000 == 0:
            print(f"Iteration #{_}")
        env = nim.NimEnv([random.randint(5, 9) for _ in range(4)])

        last_move = None

        while env.winner is None:
            state_before = copy.deepcopy(env.piles)
            action = agent.choose_action_with_epsilion(env)
            assert env.move(action) != -1
            state_after = copy.deepcopy(env.piles)

            if env.winner is not None:
                agent.update(state_before, action, state_after, -1)
                if last_move is not None:
                    agent.update(*last_move, 1)
            else:
                agent.update(state_before, action, state_after, 0)

            last_move = (state_before, action, state_after)
    agent.save()
