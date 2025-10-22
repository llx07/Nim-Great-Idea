import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import copy
import matplotlib.pyplot as plt


# Allow running this file separately
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import core.nim as nim
import agents.rand as rand


class Network(nn.Module):
    """A simple network for DQN"""

    def __init__(self, n_features: int, n_actions: int, n_hidden: int = 128):
        super(Network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=n_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=n_hidden, out_features=n_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=n_hidden, out_features=n_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=n_hidden, out_features=n_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=n_hidden, out_features=n_actions, bias=True),
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

        # set the device for training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device}")

        self.n_piles = n_piles
        self.n_stones = n_stones
        self.n_features = n_piles  # The size of feature
        self.n_actions = self.n_piles * self.n_stones  # The size of action

        self.learning_rate = 0.001  # learning rate in the optimizer
        self.gamma = 0.95  # discount factor in Q Learning

        self.replace_target_iter = 50
        self.memory_size = 5000
        self.batch_size = 32

        self.epsilon = 1  # initial epsilon
        self.epsilon_decay = 0.9995  # decay per episode
        self.min_epsilon = 0.05  # minimum epsilon

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))

        # initialize eval_net and target_net
        self.eval_net = Network(
            n_features=self.n_features, n_actions=self.n_actions
        ).to(self.device)
        if os.path.exists("model/save_eval.pt"):
            self.eval_net.load_state_dict(
                torch.load("model/save_eval.pt", map_location=self.device)
            )
        self.target_net = Network(
            n_features=self.n_features, n_actions=self.n_actions
        ).to(self.device)
        if os.path.exists("model/save_target.pt"):
            self.target_net.load_state_dict(
                torch.load("model/save_target.pt", map_location=self.device)
            )

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=self.learning_rate
        )

        self.loss_his = []

    def update_epsilon(self):
        """Update self.epsilon.

        Called when every episode ends.
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

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

    def _get_valid_actions(self, _piles: list[int]) -> list[int]:
        valid_actions = []
        for i, cnt in enumerate(_piles):
            for x in range(1, cnt + 1):
                valid_actions.append(i * self.n_stones + x - 1)

        # check if valid_actions is empty
        # try:
        #     assert len(valid_actions)
        # except AssertionError as e:
        #     print(_piles)
        #     raise e

        return valid_actions

    def choose_action(self, _piles: list[int]) -> tuple[int, int]:
        """Choose action with maximum Q-value."""

        piles = np.array(_piles)  # convert list to np.ndarray
        piles = piles[np.newaxis, :]  # add a new dimension
        s = torch.FloatTensor(piles).to(self.device)

        # use the eval_net to get all q values
        actions_value = self.eval_net(s).cpu().detach().numpy()[0]

        # get valid actions
        valid_actions = self._get_valid_actions(_piles)
        # only pick valid values
        valid_values = actions_value[valid_actions]
        # calculate real action
        action = valid_actions[np.argmax(valid_values)]

        # return in format (idx, cnt)
        idx = int(action) // self.n_stones
        cnt = int(action) % self.n_stones + 1
        return (idx, cnt)

    def choose_action_with_episilon(self, _piles: list[int]) -> tuple[int, int]:
        """Choose action with maximum Q-value, and randomly select
        action with a probability of epsilon.
        """

        if np.random.uniform() > self.epsilon:
            return self.choose_action(_piles)

        # randomly choose action with probability self.epsilon
        action = random.choice(self._get_valid_actions(_piles))
        idx = int(action) // self.n_stones
        cnt = int(action) % self.n_stones + 1
        return (idx, cnt)

    def _copy_network(self):
        """Copy the parameter of eval_net to tatget_net."""
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def _build_valid_mask(self, counts: np.ndarray):
        """Build a Tensor of shape (batch_size, n_features) according to
        counts.
        """
        # counts: shape (batch_size, n_features)
        counts = counts.astype(int)  # convert to int
        mask = np.zeros((counts.shape[0], self.n_actions), dtype=np.bool_)
        for b in range(counts.shape[0]):
            for i, cnt in enumerate(counts[b]):
                # if cnt == 0 no valid action in that pile
                for x in range(1, cnt + 1):
                    mask[b, i * self.n_stones + (x - 1)] = True
        return mask

    def learn(self):
        """Use self.memory to update the DQN net."""

        # Replace target parameters per self.replace_target_iter steps
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._copy_network()

        # sample batch memory from all memory
        batch_memory = (
            self.memory[
                np.random.choice(self.memory_size, self.batch_size, replace=False)
            ]
            if self.memory_counter > self.memory_size
            else self.memory[
                np.random.choice(self.memory_counter, self.batch_size, replace=False)
            ]
        )

        # eval the network
        s = torch.FloatTensor(batch_memory[:, : self.n_features]).to(self.device)
        ss = torch.FloatTensor(batch_memory[:, -self.n_features :]).to(self.device)
        q_eval: torch.Tensor = self.eval_net(s)
        q_next: torch.Tensor = self.target_net(ss)

        q_target = q_eval.clone()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        # valid_mask: (B, n_actions), bool
        valid_mask = self._build_valid_mask(batch_memory[:, -self.n_features :])
        valid_mask = torch.tensor(valid_mask, device=self.device)

        q_next_detached = q_next.detach().to(self.device)
        q_masked = q_next_detached.masked_fill(~valid_mask, float("-inf"))
        # get max of every row
        max_q_next = q_masked.max(dim=1)[0]
        max_q_next = torch.where(
            max_q_next == float("-inf"), torch.zeros_like(max_q_next), max_q_next
        )

        q_target[batch_index, eval_act_index] = (
            torch.FloatTensor(reward).to(self.device) + self.gamma * max_q_next
        )

        # train eval network
        loss: torch.Tensor = self.loss_function(q_target, q_eval)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1

        self.loss_his.append(loss.cpu().detach().numpy())

    def save(self):
        """Save eval_net and target_net into file."""
        torch.save(self.eval_net.state_dict(), "model/save_eval.pt")
        torch.save(self.target_net.state_dict(), "model/save_target.pt")

    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.loss_his)), self.loss_his)
        plt.show()


class DQNAgent(nim.Agent):
    def __init__(self) -> None:
        self.net = DeepQNetwork(4, 9)

    def train(self) -> None:
        step = 0

        for episode in range(1, 10001):
            if episode % 200 == 0:
                print(
                    f"Training episode #{episode}, loss = {self.net.loss_his[-1]}, ε={self.net.epsilon}"
                )
                win_rate = nim.play(self, rand.RandomAgent(), 500)[0]
                print(f"\twin rate = {win_rate*100:.2f}%")

                self.net.save()
            env = nim.NimEnv([random.randint(1, 9) for _ in range(4)])

            last = None

            for _ in range(1000):  # at most run 1000 steps
                pile_before = copy.deepcopy(env.piles)
                action = self.net.choose_action_with_episilon(pile_before)
                status = env.move(action)

                # There must be no invalid actions
                assert status >= 0

                # calculate reward
                if status == 0:
                    # Lose. give penalty
                    reward = -5.0
                    if last is not None:
                        # give reward for last transition
                        self.net.store_transition(last[0], last[1], 5.0, last[2])
                else:
                    reward = 0

                pile_after = copy.deepcopy(env.piles)
                if reward == 0:
                    if random.random() < 0.05:
                        self.net.store_transition(pile_before, action, reward, pile_after)

                if step > 200:  # only learn after 200 steps
                    self.net.learn()
                # the game ends, exit the episode
                if env.winner is not None:
                    break
                step += 1

                # store the last transiton
                last = (pile_before, action, pile_after)

            # update epsilon per episode
            self.net.update_epsilon()

        # plot the loss function
        self.net.plot_cost()

    def choose_action(self, state: nim.NimEnv) -> tuple[int, int]:
        action = self.net.choose_action(state.piles)
        available_actions = state.get_available_actions()
        # make sure the action is valid (again)
        if action not in available_actions:
            return random.choice(available_actions)
        else:
            return action


if __name__ == "__main__":
    agent = DQNAgent()
    agent.train()
