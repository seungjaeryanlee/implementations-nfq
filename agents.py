"""Reinforcement learning agents."""
import math
from typing import Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NFQAgent:
    def __init__(self, nfq_net: nn.Module, optimizer: optim.Optimizer):
        """
        Neural Fitted Q-Iteration agent.

        Parameters
        ----------
        nfq_net : nn.Module
            The Q-Network that returns estimated cost given observation and action.
        optimizer : optim.Optimzer
            Optimizer for training the NFQ network.

        """
        self._nfq_net = nfq_net
        self._optimizer = optimizer

    def get_best_action(self, obs: np.array) -> int:
        """
        Return best action for given observation according to the neural network.

        Parameters
        ----------
        obs : np.array
            An observation to find the best action for.

        Returns
        -------
        action : int
            The action chosen by greedy selection.

        """
        q_left = self._nfq_net(
            torch.cat([torch.FloatTensor(obs), torch.FloatTensor([0])], dim=0)
        )
        q_right = self._nfq_net(
            torch.cat([torch.FloatTensor(obs), torch.FloatTensor([1])], dim=0)
        )

        # Best action has lower "Q" value since it estimates cumulative cost.
        return 1 if q_left >= q_right else 0

    # TODO(seungjaeryanlee): Move to environment
    def get_goal_patterns(self, factor=100):
        """Use hint-to-goal heuristic to clamp network output."""
        goal_patterns = []
        for _ in range(factor):
            state_action_pair = np.array(
                [
                    # TODO(seungjaeryanlee): What is goal velocity?
                    np.random.uniform(-0.05, 0.05),
                    np.random.normal(),
                    np.random.uniform(-math.pi / 2, math.pi / 2),
                    np.random.normal(),
                    np.random.randint(2),
                ]
            )
            goal_patterns.append(state_action_pair)

        return goal_patterns

    def train(self, rollout, gamma=0.95):
        """Train neural network with a given rollout."""
        state_batch, action_batch, cost_batch, next_state_batch, done_batch = zip(
            *rollout
        )
        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.FloatTensor(action_batch)
        cost_batch = torch.FloatTensor(cost_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch)

        state_action_batch = torch.cat([state_batch, action_batch.unsqueeze(1)], 1)
        predicted_q_values = self._nfq_net(state_action_batch).squeeze()

        # Compute min_a Q(s', a)
        q_next_state_left_batch = self._nfq_net(
            torch.cat([next_state_batch, torch.zeros(len(rollout), 1)], 1)
        ).squeeze()
        q_next_state_right_batch = self._nfq_net(
            torch.cat([next_state_batch, torch.ones(len(rollout), 1)], 1)
        ).squeeze()
        q_next_state_batch = torch.min(
            q_next_state_left_batch, q_next_state_right_batch
        )

        # TODO(seungjaeryanlee): Done mask not mentioned in paper, but should I add it?
        with torch.no_grad():
            target_q_values = cost_batch + gamma * q_next_state_batch

        # Variant 2: Clamp function to zero in goal region
        goal_patterns = self.get_goal_patterns(factor=100)
        goal_patterns = torch.FloatTensor(goal_patterns)
        predicted_goal_values = self._nfq_net(goal_patterns).squeeze()
        goal_target = torch.FloatTensor([0] * 100)
        predicted_q_values = torch.cat(
            [predicted_q_values, predicted_goal_values], dim=0
        )
        target_q_values = torch.cat([target_q_values, goal_target], dim=0)

        loss = F.mse_loss(predicted_q_values, target_q_values)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def evaluate(self, eval_env: gym.Env) -> Tuple[int, str]:
        """Evaluate NFQ agent on evaluation environment.

        Parameters
        ----------
        eval_env : gym.Env
            Environment to evaluate the agent.

        Returns
        -------
        steps : int
            Number of steps the agent took.
        success : bool
            True if the agent was terminated due to max timestep.

        """
        steps = 0
        obs = eval_env.reset()
        done = False

        while not done:
            action = self.get_best_action(obs)
            obs, _, done, info = eval_env.step(action)
            steps += 1

        return steps, steps == eval_env.max_steps
