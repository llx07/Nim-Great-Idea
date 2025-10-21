import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random

import core.nim as nim


class Network(nn.Module):
    """A simple network for DQN"""

    def __init__(self, n_features: int, n_actions: int, n_hidden: int = 20):
        super(Network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=n_hidden, bias=True),
            nn.Linear(in_features=n_hidden, out_features=n_actions, bias=True),
            nn.ReLU(),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        q = self.net(s)
        return q


class DeepQNetwork(nn.Module):
    """Q Learning with NN."""

    def __init__(self, n_piles: int, n_stones: int):
        """Init the DeepQNetwork.

        Parameters:
            n_piles: the number of piles
            n_stones: max number of stones in one pile
        """
        super(DeepQNetwork, self).__init__()

        self.n_piles = n_piles
        self.n_stones = n_stones
        self.n_features = n_piles  # The size of feature
        self.n_actions = self.n_piles * self.n_stones  # The size of action

        self.learning_rate = 0.01
        self.gamma = 0.9

        self.replace_target_iter = 300
        self.memory_size = 500
        self.batch_size = 200

        self.epsilon = 0.1

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))

        self.eval_net = Network(n_features=self.n_features, n_actions=self.n_actions)
        if os.path.exists("save_eval.pt"):
            self.eval_net.load_state_dict(torch.load("save_eval.pt"))

        self.target_net = Network(n_features=self.n_features, n_actions=self.n_actions)
        if os.path.exists("save_target.pt"):
            self.target_net.load_state_dict(torch.load("save_target.pt"))

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=self.learning_rate
        )

    def store_transition(
        self, s: list[int], _a: tuple[int, int], r: float, ss: list[int]
    ):
        """Store transiton (s, a, r, ss) into memory.

        Parameters:
            s: the old state.
            _a: the action, in format (idx, cnt).
            r: the reward.
            ss: the new state.
        """
        idx, cnt = _a
        a = idx * self.n_stones + cnt - 1
        transition = np.hstack((s, [a, r], ss))
        index = self.memory_counter % self.memory_size
        self.memory[index] = transition
        self.memory_counter += 1

    def choose_action(self, _piles: list[int]) -> tuple[int, int]:
        piles = np.array(_piles)  # convert list to np.ndarray
        piles = piles[np.newaxis, :]  # add a new dimension
        if np.random.uniform() > self.epsilon:
            # forward feed the observation and get q value for every actions
            s = torch.FloatTensor(piles)
            actions_value: torch.Tensor = self.eval_net(s)
            action = [np.argmax(actions_value.detach().numpy())][0]
        else:
            action = np.random.randint(0, self.n_actions)
        idx = int(action) // self.n_stones
        cnt = int(action) % self.n_stones + 1
        return (idx, cnt)

    def _copy_network(self):
        """Copy the parameter of eval_net to tatget_net."""
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def learn(self):
        # Replace targe parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._copy_network()
            print("target params replaced")

        # sample batch memory from all memory
        batch_memory = (
            np.random.choice(self.memory, self.batch_size, replace=False)
            if self.memory_counter > self.memory_size
            else np.random.choice(
                self.memory[: self.memory_counter], self.batch_size, replace=True
            )
        )

        # run the nextwork
        s = torch.FloatTensor(batch_memory[:, : self.n_features])
        ss = torch.FloatTensor(batch_memory[:, : self.n_features])
        q_eval: torch.Tensor = self.eval_net(s)
        q_next: torch.Tensor = self.target_net(ss)

        # change q_target w.r.t q_eval's action
        q_target = q_eval.clone()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = (
            torch.FloatTensor(reward) + self.gamma * q_next.max(dim=1).values
        )

        # train eval network
        loss: torch.Tensor = self.loss_function(q_target, q_eval)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1

    def save(self):
        """Save eval_net and target_net into file."""
        torch.save(self.eval_net, "save_eval.pt")
        torch.save(self.target_net, "save_target.pt")


class DQNAgent(nim.Agent):
    def __init__(self) -> None:
        self.net = DeepQNetwork(4, 9)

    def train(self) -> None:
        step = 0
        for episode in range(200): # train for 200 episodes
            print(f"Training episode #{episode}")
            env = nim.NimEnv([random.randint(0, 9) for _ in range(4)])
            for _ in range(100):  # at most run 100 steps
                pile_before = env.piles
                action = self.net.choose_action(pile_before)
                try:
                    reward = env.move(action)
                except ValueError:
                    reward = -100.0
                pile_after = env.piles

                self.net.store_transition(pile_before, action, reward, pile_after)

                if step > 200 and step % 5 == 0:
                    self.net.learn()

                # the game ends, exit the episode
                if env.winner is not None:
                    break

                step += 1

    def choose_action(self, state: nim.NimEnv) -> tuple[int, int]:
        action = self.net.choose_action(state.piles)
        idx, cnt = action

        if cnt > state.piles[idx]:
            return random.choice(state.get_available_actions())
        else:
            return action

if __name__ == "__main__":
    agent = DQNAgent()
    agent.train()
