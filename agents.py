"""Reinforcement learning agents."""
import numpy as np
import torch
import torch.nn as nn


class NFQAgent:
    def __init__(self, nfq_net: nn.Module):
        """
        Neural Fitted Q-Iteration agent.

        Parameters
        ----------
        nfq_net : nn.Module
            The Q-Network that returns estimated cost given observation and action.

        """
        self._nfq_net = nfq_net

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

    def evaluate(self, eval_env, episodes=1):
        """Evaluate NFQ agent on test environment."""
        steps = 0
        nb_success = 0
        for _ in range(episodes):
            obs = eval_env.reset()
            done = False

            while not done:
                action = self.get_best_action(obs)
                obs, _, done, info = eval_env.step(action)
                steps += 1

            nb_success += 1 if info["state"] == "success" else 0

        avg_number_of_steps = float(steps) / episodes
        success_rate = float(nb_success) / episodes

        return avg_number_of_steps, success_rate
