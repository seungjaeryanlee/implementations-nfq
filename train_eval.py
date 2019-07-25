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
import os
from typing import Callable, List, Tuple

import configargparse
import gym
import numpy as np
import torch
import torch.optim as optim

from agents import NFQAgent
from environments import CartPoleRegulatorEnv
from networks import NFQNetwork
from utils import get_logger, make_reproducible


def generate_rollout(
    env: gym.Env, get_best_action: Callable = None, render: bool = False
) -> List[Tuple[np.array, int, int, np.array, bool]]:
    """
    Generate rollout using given neural network.

    If a network is not given, generate random rollout instead.

    Parameters
    ----------
    env : gym.Env
        The environment to generate rollout from.
    get_best_action : Callable
        Greedy policy.
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
        action = get_best_action(obs) if get_best_action else env.action_space.sample()
        next_obs, cost, done, _ = env.step(action)
        rollout.append((obs, action, cost, next_obs, done))
        obs = next_obs

        if render:
            env.render()

    return rollout


def main():
    """Run NFQ."""
    # Setup hyperparameters
    parser = configargparse.ArgParser()
    parser.add("-c", "--config", required=True, is_config_file=True)
    parser.add("--EPOCH", type=int)
    parser.add("--TRAIN_ENV_MAX_STEPS", type=int)
    parser.add("--EVAL_ENV_MAX_STEPS", type=int)
    parser.add("--DISCOUNT", type=float)
    parser.add("--INIT_EXPERIENCE", type=int)
    parser.add("--INCREMENT_EXPERIENCE", action="store_true")
    parser.add("--HINT_TO_GOAL", action="store_true")
    parser.add("--RANDOM_SEED", type=int)
    parser.add("--SAVE_PATH", type=str, default="")
    parser.add("--LOAD_PATH", type=str, default="")
    parser.add("--USE_TENSORBOARD", action="store_true")
    parser.add("--USE_WANDB", action="store_true")
    CONFIG = parser.parse_args()
    if not hasattr(CONFIG, "INCREMENT_EXPERIENCE"):
        CONFIG.USE_TENSORBOARD = False
    if not hasattr(CONFIG, "HINT_TO_GOAL"):
        CONFIG.USE_WANDB = False
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
    eval_env = CartPoleRegulatorEnv(mode="eval")

    # Fix random seeds
    if CONFIG.RANDOM_SEED is not None:
        make_reproducible(
            CONFIG.RANDOM_SEED, use_random=True, use_numpy=True, use_torch=True
        )
        train_env.seed(CONFIG.RANDOM_SEED)
        eval_env.seed(CONFIG.RANDOM_SEED)
    else:
        logger.warning("Running without a random seed: this run is NOT reproducible.")

    # Setup agent
    nfq_net = NFQNetwork()
    optimizer = optim.Rprop(nfq_net.parameters())
    nfq_agent = NFQAgent(nfq_net, optimizer)

    # Load trained agent
    if CONFIG.LOAD_PATH:
        state_dict = torch.load(CONFIG.LOAD_PATH)
        nfq_net.load_state_dict(state_dict["nfq_net"])
        optimizer.load_state_dict(state_dict["optimizer"])

    # NFQ Main loop
    # A set of transition samples denoted as D
    all_rollouts = []
    if CONFIG.INIT_EXPERIENCE:
        for _ in range(CONFIG.INIT_EXPERIENCE):
            all_rollouts.extend(generate_rollout(train_env, None, render=False))
    for epoch in range(CONFIG.EPOCH + 1):
        # Variant 1: Incermentally add transitions (Section 3.4)
        # TODO(seungjaeryanlee): Done before or after training?
        if CONFIG.INCREMENT_EXPERIENCE:
            new_rollout = generate_rollout(
                train_env, nfq_agent.get_best_action, render=False
            )
            all_rollouts.extend(new_rollout)

        state_action_b, target_q_values = nfq_agent.generate_pattern_set(all_rollouts)

        # Variant 2: Clamp function to zero in goal region
        if CONFIG.HINT_TO_GOAL:
            goal_state_action_b, goal_target_q_values = train_env.get_goal_pattern_set()
            goal_state_action_b = torch.FloatTensor(goal_state_action_b)
            goal_target_q_values = torch.FloatTensor(goal_target_q_values)
            state_action_b = torch.cat([state_action_b, goal_state_action_b], dim=0)
            target_q_values = torch.cat([target_q_values, goal_target_q_values], dim=0)

        nfq_agent.train((state_action_b, target_q_values))
        eval_score, eval_success = nfq_agent.evaluate(eval_env)

        logger.info(
            "Epoch {:4d} | Rollout Steps: {:4d} | Evaluation Steps: {:4d}".format(
                epoch, len(new_rollout), eval_score
            )
        )
        if CONFIG.USE_TENSORBOARD:
            writer.add_scalar("train/episode_length", len(new_rollout), epoch)
            writer.add_scalar("eval/episode_length", eval_score, epoch)
        if CONFIG.USE_WANDB:
            wandb.log({"Train Episode Length": len(new_rollout)}, step=epoch)
            wandb.log({"Evaluation Episode Length": eval_score}, step=epoch)

        if eval_success:
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
    eval_env.close()


if __name__ == "__main__":
    main()
