"""Reinforcement learning agents."""
from typing import List, Tuple

import gym
import numpy as np
import tensorflow as tf


class NFQAgent:
    def __init__(self, nfq_net: tf.keras.Model, optimizer: tf.keras.optimizers):
        """
        Neural Fitted Q-Iteration agent.

        Parameters
        ----------
        nfq_net : tf.keras.Model
            The Q-Network that returns estimated cost given observation and action.
        optimizer : tf.keras.optimizers
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
        np_obs_left = np.expand_dims(np.concatenate([obs, [0]], axis=0), axis=0)
        np_obs_right = np.expand_dims(np.concatenate([obs, [1]], axis=0), axis=0)
        assert np_obs_left.shape == (1, 5)
        assert np_obs_right.shape == (1, 5)

        q_left = self._nfq_net(tf.convert_to_tensor(np_obs_left))
        q_right = self._nfq_net(tf.convert_to_tensor(np_obs_right))
        assert q_left.shape == (1, 1)
        assert q_right.shape == (1, 1)

        # Best action has lower "Q" value since it estimates cumulative cost.
        return 1 if q_left[0][0] >= q_right[0][0] else 0

    def generate_pattern_set(
        self,
        rollouts: List[Tuple[np.array, int, int, np.array, bool]],
        gamma: float = 0.95,
    ) -> Tuple[np.array, np.array]:
        """Generate pattern set.

        Parameters
        ----------
        rollouts : list of tuple
            Generated rollouts, which is a tuple of state, action, cost, next state, and done.
        gamma : float
            Discount factor. Defaults to 0.95.

        Returns
        -------
        pattern_set : tuple of np.array
            Pattern set to train the NFQ network.

        """
        # _b denotes batch
        state_b, action_b, cost_b, next_state_b, done_b = zip(*rollouts)
        state_b = np.array(state_b)
        action_b = np.expand_dims(np.array(action_b), axis=1)
        cost_b = np.array(cost_b)
        next_state_b = np.array(next_state_b)
        done_b = np.array(done_b)
        assert state_b.shape == (len(rollouts), 4)
        assert action_b.shape == (len(rollouts), 1)
        assert cost_b.shape == (len(rollouts),)
        assert next_state_b.shape == (len(rollouts), 4)
        assert done_b.shape == (len(rollouts),)

        state_action_b = np.concatenate([state_b, action_b], axis=1)
        assert state_action_b.shape == (len(rollouts), 5)

        # Compute Q(s', 0) and Q(s', 1)
        next_state_left_b = np.concatenate(
            [next_state_b, np.expand_dims(np.zeros(len(rollouts)), axis=1)], axis=1
        )
        next_state_right_b = np.concatenate(
            [next_state_b, np.expand_dims(np.ones(len(rollouts)), axis=1)], axis=1
        )
        assert next_state_left_b.shape == (len(rollouts), 5)
        assert next_state_right_b.shape == (len(rollouts), 5)
        q_next_state_left_b = self._nfq_net(tf.convert_to_tensor(next_state_left_b))
        q_next_state_right_b = self._nfq_net(tf.convert_to_tensor(next_state_right_b))
        assert q_next_state_left_b.shape == (len(rollouts), 1)
        assert q_next_state_right_b.shape == (len(rollouts), 1)

        # Compute min_a Q(s', a)
        q_next_state_b = (
            tf.minimum(q_next_state_left_b, q_next_state_right_b).numpy().flatten()
        )
        assert q_next_state_b.shape == (len(rollouts),)

        # If goal state (S+): target = 0 + gamma * min Q
        # If forbidden state (S-): target = 1
        # If neither: target = c_trans + gamma * min Q
        # NOTE(seungjaeryanlee): done is True only when the episode terminated
        #                        due to entering forbidden state. It is not
        #                        True if it terminated due to maximum timestep.
        # TODO TensorFlow
        # with torch.no_grad():
        target_q_values = cost_b + gamma * q_next_state_b * (1 - done_b)
        assert target_q_values.shape == (len(rollouts),)

        return state_action_b, target_q_values

    def train(self, pattern_set: Tuple[tf.Tensor, tf.Tensor]) -> float:
        """Train neural network with a given pattern set.

        Parameters
        ----------
        pattern_set : tuple of tf.Tensor
            Pattern set to train the NFQ network.

        Returns
        -------
        loss : float
            Training loss.

        """
        state_action_b, target_q_values = pattern_set
        target_q_values = np.expand_dims(target_q_values, axis=1)

        with tf.GradientTape() as tape:
            predicted_q_values = self._nfq_net(state_action_b)
            # TODO(seungjaeryanlee): tf.keras.losses.MSE doesn't work... Any better way?
            loss = tf.reduce_mean((predicted_q_values - target_q_values) ** 2)

        grads = tape.gradient(loss, self._nfq_net.trainable_weights)
        self._optimizer.apply_gradients(zip(grads, self._nfq_net.trainable_weights))

        return loss.numpy()

    def evaluate(self, eval_env: gym.Env, render: bool) -> Tuple[int, str, float]:
        """Evaluate NFQ agent on evaluation environment.

        Parameters
        ----------
        eval_env : gym.Env
            Environment to evaluate the agent.
        render: bool
            If true, render environment.

        Returns
        -------
        episode_length : int
            Number of steps the agent took.
        success : bool
            True if the agent was terminated due to max timestep.
        episode_cost : float
            Total cost accumulated from the evaluation episode.

        """
        episode_length = 0
        obs = eval_env.reset()
        done = False
        info = {"time_limit": False}
        episode_cost = 0
        while not done and not info["time_limit"]:
            action = self.get_best_action(obs)
            obs, cost, done, info = eval_env.step(action)
            episode_cost += cost
            episode_length += 1

            if render:
                eval_env.render()

        success = (
            episode_length == eval_env.max_steps
            and abs(obs[0]) <= eval_env.x_success_range
        )

        return episode_length, success, episode_cost
