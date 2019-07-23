#!/usr/bin/env python
"""Implement Neural Fitted Q-Iteration.

http://ml.informatik.uni-freiburg.de/former/_media/publications/rieecml05.pdf


Running
-------
You can train the NFQ agent on CartPole Regulator with the inluded
configuration file with the below command:
```
python train_eval.py -c cartpole.conf
```

For a reproducible run, use the RANDOM_SEED flag.
```
python train_eval.py -c cartpole.conf --RANDOM_SEED=1
```

To save a trained agent, use the SAVE_PATH flag.
```
python train_eval.py -c cartpole.conf --SAVE_PATH=saves/cartpole.pth
```

To load a trained agent, use the LOAD_PATH flag.
```
python train_eval.py -c cartpole.conf --LOAD_PATH=saves/cartpole.pth
```

To enable logging to TensorBoard or W&B, use appropriate flags.
```
python train_eval.py -c cartpole.conf --USE_TENSORBOARD --USE_WANDB
```


Logging
-------
1. You can view runs online via Weights & Biases (wandb):
https://app.wandb.ai/seungjaeryanlee/implementations-nfq/runs

2. You can use TensorBoard to view runs offline:
```
tensorboard --logdir=tensorboard_logs --port=2223
```


Glossary
--------
env : Environment
obs : Observation
"""
import math
import os
from typing import List, Tuple

import configargparse
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from cartpole import CartPoleRegulatorEnv
from networks import NFQNetwork
from utils import get_logger, make_reproducible


def get_best_action(nfq_net: nn.Module, obs: np.array) -> int:
    """
    Return best action for given observation according to the neural network.

    Parameters
    ----------
    nfq_net : nn.Module
        The Q-Network that returns estimated cost given observation and action.
    obs : np.array
        An observation to find the best action for.

    Returns
    -------
    action : int
        The action chosen by greedy selection.

    """
    q_left = nfq_net(torch.cat([torch.FloatTensor(obs), torch.FloatTensor([0])], dim=0))
    q_right = nfq_net(
        torch.cat([torch.FloatTensor(obs), torch.FloatTensor([1])], dim=0)
    )

    # Best action has lower "Q" value since it estimates cumulative cost.
    return 1 if q_left >= q_right else 0


def generate_rollout(
    env: gym.Env, nfq_net: nn.Module = None, render: bool = False
) -> List[Tuple[np.array, int, int, np.array, bool]]:
    """
    Generate rollout using given neural network.

    If a network is not given, generate random rollout instead.

    Parameters
    ----------
    env : gym.Env
        The environment to generate rollout from.
    nfq_net : nn.Module
        The Q-Network that returns estimated cost given observation and action.
    render: bool
        If true, render environment.

    Returns
    -------
    rollout : List of Tuple
        Generated rollout.

    """
    rollout = []
    obs = env.reset()
    done = False
    while not done:
        action = get_best_action(nfq_net, obs) if nfq_net else env.action_space.sample()
        next_obs, cost, done, _ = env.step(action)
        rollout.append((obs, action, cost, next_obs, done))
        obs = next_obs

        if render:
            env.render()

    return rollout


def train(nfq_net, optimizer, rollout, gamma=0.95):
    """Train neural network with a given rollout."""
    state_batch, action_batch, cost_batch, next_state_batch, done_batch = zip(*rollout)
    state_batch = torch.FloatTensor(state_batch)
    action_batch = torch.FloatTensor(action_batch)
    cost_batch = torch.FloatTensor(cost_batch)
    next_state_batch = torch.FloatTensor(next_state_batch)
    done_batch = torch.FloatTensor(done_batch)

    state_action_batch = torch.cat([state_batch, action_batch.unsqueeze(1)], 1)
    predicted_q_values = nfq_net(state_action_batch).squeeze()

    # Compute min_a Q(s', a)
    q_next_state_left_batch = nfq_net(
        torch.cat([next_state_batch, torch.zeros(len(rollout), 1)], 1)
    ).squeeze()
    q_next_state_right_batch = nfq_net(
        torch.cat([next_state_batch, torch.ones(len(rollout), 1)], 1)
    ).squeeze()
    q_next_state_batch = torch.min(q_next_state_left_batch, q_next_state_right_batch)

    # TODO(seungjaeryanlee): Done mask not mentioned in paper, but should I add it?
    with torch.no_grad():
        target_q_values = cost_batch + gamma * q_next_state_batch

    # Variant 2: Clamp function to zero in goal region
    goal_patterns = get_goal_patterns(nfq_net, optimizer, factor=100)
    goal_patterns = torch.FloatTensor(goal_patterns)
    predicted_goal_values = nfq_net(goal_patterns).squeeze()
    goal_target = torch.FloatTensor([0] * 100)
    predicted_q_values = torch.cat([predicted_q_values, predicted_goal_values], dim=0)
    target_q_values = torch.cat([target_q_values, goal_target], dim=0)

    loss = F.mse_loss(predicted_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def get_goal_patterns(nfq_net, optimizer, factor=100):
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


def test(env, nfq_net, episodes=1):
    """Test NFQ agent on test environment."""
    steps = 0
    nb_success = 0
    for _ in range(episodes):
        obs = env.reset()
        done = False

        while not done:
            action = get_best_action(nfq_net, obs)
            obs, _, done, info = env.step(action)
            steps += 1

        nb_success += 1 if info["state"] == "success" else 0

    avg_number_of_steps = float(steps) / episodes
    success_rate = float(nb_success) / episodes

    return avg_number_of_steps, success_rate


def main():
    """Run NFQ."""
    # Setup hyperparameters
    parser = configargparse.ArgParser()
    parser.add("-c", "--config", required=True, is_config_file=True)
    parser.add("--EPOCH", dest="EPOCH", type=int)
    parser.add("--TRAIN_ENV_MAX_STEPS", dest="TRAIN_ENV_MAX_STEPS", type=int)
    parser.add("--TEST_ENV_MAX_STEPS", dest="TEST_ENV_MAX_STEPS", type=int)
    parser.add("--DISCOUNT", dest="DISCOUNT", type=float)
    parser.add("--RANDOM_SEED", dest="RANDOM_SEED", type=int)
    parser.add("--SAVE_PATH", dest="SAVE_PATH", type=str, default="")
    parser.add("--LOAD_PATH", dest="LOAD_PATH", type=str, default="")
    parser.add("--USE_TENSORBOARD", dest="USE_TENSORBOARD", action="store_true")
    parser.add("--USE_WANDB", dest="USE_WANDB", action="store_true")
    CONFIG = parser.parse_args()
    if not hasattr(CONFIG, "USE_TENSORBOARD"):
        CONFIG.USE_TENSORBOARD = False
    if not hasattr(CONFIG, "USE_WANDB"):
        CONFIG.USE_WANDB = False

    print()
    print("+--------------------------------+--------------------------------+")
    print("| Hyperparameters                | Value                          |")
    print("+--------------------------------+--------------------------------+")
    for arg in vars(CONFIG):
        print(
            "| {:30} | {:<30} |".format(
                arg, getattr(CONFIG, arg) if getattr(CONFIG, arg) is not None else ""
            )
        )
    print("+--------------------------------+--------------------------------+")
    print()

    # Log to File, Console, TensorBoard, W&B
    logger = get_logger()

    if CONFIG.USE_TENSORBOARD:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir="tensorboard_logs")
    if CONFIG.USE_WANDB:
        import wandb

        wandb.init(project="implementations-nfq", config=CONFIG)

    # Setup environment
    train_env = CartPoleRegulatorEnv(mode="train")
    test_env = CartPoleRegulatorEnv(mode="test")

    # Fix random seeds
    if CONFIG.RANDOM_SEED is not None:
        make_reproducible(
            CONFIG.RANDOM_SEED, use_random=True, use_numpy=True, use_torch=True
        )
        train_env.seed(CONFIG.RANDOM_SEED)
        test_env.seed(CONFIG.RANDOM_SEED)
    else:
        logger.warning("Running without a random seed: this run is NOT reproducible.")

    # Setup agent
    nfq_net = NFQNetwork()
    optimizer = optim.Rprop(nfq_net.parameters())

    # Load trained agent
    if CONFIG.LOAD_PATH:
        state_dict = torch.load(CONFIG.LOAD_PATH)
        nfq_net.load_state_dict(state_dict["nfq_net"])
        optimizer.load_state_dict(state_dict["optimizer"])

    rollout = []
    for epoch in range(CONFIG.EPOCH + 1):
        # Variant 1: Incermentally add transitions (Section 3.4)
        new_rollout = generate_rollout(train_env, nfq_net, render=False)
        rollout.extend(new_rollout)

        logger.info(
            "Epoch {:4d} | TRAINING   | Steps: {:4d}".format(epoch, len(new_rollout))
        )
        if CONFIG.USE_TENSORBOARD:
            writer.add_scalar("train/episode_length", len(new_rollout), epoch)
        if CONFIG.USE_WANDB:
            wandb.log({"Train Episode Length": len(new_rollout)}, step=epoch)

        # Train from all past experience
        train(nfq_net, optimizer, rollout)

        # Test on 3000-step environment
        number_of_steps, _ = test(test_env, nfq_net, episodes=1)
        logger.info(
            "Epoch {:4d} | TEST       | Steps: {:4d}".format(
                epoch, int(number_of_steps)
            )
        )
        if CONFIG.USE_TENSORBOARD:
            writer.add_scalar("test/episode_length", int(number_of_steps), epoch)
        if CONFIG.USE_WANDB:
            wandb.log({"Test Episode Length": int(number_of_steps)}, step=epoch)

        if number_of_steps == 3000:
            break

    # Save trained agent
    if CONFIG.SAVE_PATH:
        # Create specified directory if it does not exist yet
        SAVE_DIRECTORY = "/".join(CONFIG.SAVE_PATH.split("/")[:-1])
        if not os.path.exists(SAVE_DIRECTORY):
            os.makedirs(SAVE_DIRECTORY)

        torch.save(
            {"nfq_net": nfq_net.state_dict(), "optimizer": optimizer.state_dict()},
            CONFIG.SAVE_PATH,
        )

    train_env.close()
    test_env.close()


if __name__ == "__main__":
    main()
