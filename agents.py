"""Reinforcement learning agents."""
import math
from typing import List, Tuple

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
    def get_goal_pattern_set(
        self, size: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use hint-to-goal heuristic to clamp network output.

        Parameters
        ----------
        size : int
            The size of the goal pattern set to generate.

        Returns
        -------
        pattern_set : tuple of torch.Tensor
            Pattern set to train the NFQ network.

        """
        goal_state_action_b = [
            np.array(
                [
                    # TODO(seungjaeryanlee): What is goal velocity?
                    np.random.uniform(-0.05, 0.05),
                    np.random.normal(),
                    np.random.uniform(-math.pi / 2, math.pi / 2),
                    np.random.normal(),
                    np.random.randint(2),
                ]
            )
            for _ in range(size)
        ]
        goal_target_q_values = [0] * size

        return goal_state_action_b, goal_target_q_values

    def generate_pattern_set(
        self,
        rollouts: List[Tuple[np.array, int, int, np.array, bool]],
        gamma: float = 0.95,
    ):
        """Generate pattern set.

        Parameters
        ----------
        rollouts : list of tuple
            Generated rollouts, which is a tuple of state, action, cost, next state, and done.
        gamma : float
            Discount factor. Defaults to 0.95.

        Returns
        -------
        pattern_set : tuple of torch.Tensor
            Pattern set to train the NFQ network.

        """
        # _b denotes batch
        state_b, action_b, cost_b, next_state_b, done_b = zip(*rollouts)
        state_b = torch.FloatTensor(state_b)
        action_b = torch.FloatTensor(action_b)
        cost_b = torch.FloatTensor(cost_b)
        next_state_b = torch.FloatTensor(next_state_b)
        done_b = torch.FloatTensor(done_b)

        state_action_b = torch.cat([state_b, action_b.unsqueeze(1)], 1)

        # Compute min_a Q(s', a)
        q_next_state_left_b = self._nfq_net(
            torch.cat([next_state_b, torch.zeros(len(rollouts), 1)], 1)
        ).squeeze()
        q_next_state_right_b = self._nfq_net(
            torch.cat([next_state_b, torch.ones(len(rollouts), 1)], 1)
        ).squeeze()
        q_next_state_b = torch.min(q_next_state_left_b, q_next_state_right_b)

        # NOTE(seungjaeryanlee): Done mask not mentioned in paper
        with torch.no_grad():
            target_q_values = cost_b + gamma * q_next_state_b * (1 - done_b)

        return state_action_b, target_q_values

    def train(self, pattern_set: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Train neural network with a given pattern set.

        Parameters
        ----------
        pattern_set : tuple of torch.Tensor
            Pattern set to train the NFQ network.

        Returns
        -------
        loss : float
            Training loss.

        """
        state_action_b, target_q_values = pattern_set

        # TODO(seungjaeryanlee): Move somewhere else?
        # Variant 2: Clamp function to zero in goal region
        goal_state_action_b, goal_target_q_values = self.get_goal_pattern_set()
        goal_state_action_b = torch.FloatTensor(goal_state_action_b)
        goal_target_q_values = torch.FloatTensor(goal_target_q_values)
        state_action_b = torch.cat([state_action_b, goal_state_action_b], dim=0)
        target_q_values = torch.cat([target_q_values, goal_target_q_values], dim=0)

        predicted_q_values = self._nfq_net(state_action_b).squeeze()
        loss = F.mse_loss(predicted_q_values, target_q_values)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()

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
