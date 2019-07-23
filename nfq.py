#!/usr/bin/env python
import random
import math

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from cartpole import CartPoleRegulatorEnv
from networks import NFQNetwork
from utils import make_reproducible, get_logger


def get_best_action(net, obs):
    """
    Return best action for given observation according to the neural network.
    Best action has lower "Q" value since it estimates cumulative cost.
    """
    obs_left = torch.cat([torch.FloatTensor(obs), torch.FloatTensor([0])], 0)
    obs_right = torch.cat([torch.FloatTensor(obs), torch.FloatTensor([1])], 0)
    q_left = net(obs_left)
    q_right = net(obs_right)
    action = 1 if q_left >= q_right else 0

    return action


def generate_rollout(env, net=None, render=False):
    """
    Generate rollout using given neural network. If a network is not given,
    generate random rollout instead.
    """
    rollout = []
    obs = env.reset()
    done = False
    while not done:
        action = get_best_action(net, obs) if net else env.action_space.sample()
        next_obs, cost, done, _ = env.step(action)
        rollout.append((obs, action, cost, next_obs, done))
        obs = next_obs

        if render:
            env.render()

    return rollout


def train(net, optimizer, rollout, gamma=0.95):
    """
    Train neural network with a given rollout.
    """
    state_batch, action_batch, cost_batch, next_state_batch, done_batch = zip(*rollout)
    state_batch = torch.FloatTensor(state_batch)
    action_batch = torch.FloatTensor(action_batch)
    cost_batch = torch.FloatTensor(cost_batch)
    next_state_batch = torch.FloatTensor(next_state_batch)
    done_batch = torch.FloatTensor(done_batch)

    state_action_batch = torch.cat([state_batch, action_batch.unsqueeze(1)], 1)
    predicted_q_values = net(state_action_batch).squeeze()

    # Compute min_a Q(s', a)
    q_next_state_left_batch = net(
        torch.cat([next_state_batch, torch.zeros(len(rollout), 1)], 1)
    ).squeeze()
    q_next_state_right_batch = net(
        torch.cat([next_state_batch, torch.ones(len(rollout), 1)], 1)
    ).squeeze()
    q_next_state_batch = torch.min(q_next_state_left_batch, q_next_state_right_batch)

    # TODO(seungjaeryanlee): Done mask not mentioned in paper, but should I add it?
    with torch.no_grad():
        target_q_values = cost_batch + gamma * q_next_state_batch

    loss = F.mse_loss(predicted_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def hint_to_goal(net, optimizer, factor=100):
    """
    Use hint-to-goal heuristic to clamp network output.
    """
    for _ in range(factor):
        state_action_pair = torch.FloatTensor(
            [
                # TODO(seungjaeryanlee): What is goal velocity?
                [
                    np.random.uniform(-0.05, 0.05),
                    np.random.normal(),
                    np.random.uniform(-math.pi, math.pi),
                    np.random.normal(),
                    random.randint(0, 1),
                ]
            ]
        )
        predicted_q_value = net(state_action_pair.flatten())
        loss = F.mse_loss(predicted_q_value, torch.FloatTensor([0]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(env, net, episodes=1000):
    steps = 0
    nb_success = 0
    for _ in range(episodes):
        obs = env.reset()
        done = False

        while not done:
            action = get_best_action(net, obs)
            obs, _, done, info = env.step(action)
            steps += 1

        nb_success += 1 if info["state"] == "success" else 0

    avg_number_of_steps = float(steps) / episodes
    success_rate = float(nb_success) / episodes

    return avg_number_of_steps, success_rate


def main():
    logger = get_logger()
    make_reproducible(0xC0FFEE, use_random=True, use_torch=True)

    train_env = CartPoleRegulatorEnv(mode="train")
    test_env = CartPoleRegulatorEnv(mode="test")
    train_env.seed(0xC0FFEE)
    test_env.seed(0xC0FFEE)

    net = NFQNetwork()
    optimizer = optim.Rprop(net.parameters())
    # TODO Initialize weights randomly within [-0.5, 0.5]

    for epoch in range(500+1):
        rollout = generate_rollout(train_env, net, render=False)
        
        logger.info("Epoch {:4d} | TRAINING   | Steps: {:3d}".format(epoch, len(rollout)))
        train(net, optimizer, rollout)
        hint_to_goal(net, optimizer)
        # avg_number_of_steps, success_rate = test(test_env, net)
        # logger.info("Epoch {:4d} | EVALUATION | AVG # Steps: {:3.3f} | Success: {:3.1f}%".format(epoch, avg_number_of_steps, success_rate))

    train_env.close()
    test_env.close()


if __name__ == "__main__":
    main()
