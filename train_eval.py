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
import configargparse
import torch
import torch.optim as optim

from environments import CartPoleRegulatorEnv
from nfq.agents import NFQAgent
from nfq.networks import NFQNetwork
from utils import get_logger, load_models, make_reproducible, save_models


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
    parser.add("--TRAIN_RENDER", action="store_true")
    parser.add("--EVAL_RENDER", action="store_true")
    parser.add("--SAVE_PATH", type=str, default="")
    parser.add("--LOAD_PATH", type=str, default="")
    parser.add("--USE_TENSORBOARD", action="store_true")
    parser.add("--USE_WANDB", action="store_true")
    CONFIG = parser.parse_args()
    if not hasattr(CONFIG, "INCREMENT_EXPERIENCE"):
        CONFIG.INCREMENT_EXPERIENCE = False
    if not hasattr(CONFIG, "HINT_TO_GOAL"):
        CONFIG.HINT_TO_GOAL = False
    if not hasattr(CONFIG, "TRAIN_RENDER"):
        CONFIG.TRAIN_RENDER = False
    if not hasattr(CONFIG, "EVAL_RENDER"):
        CONFIG.EVAL_RENDER = False
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
        make_reproducible(CONFIG.RANDOM_SEED, use_numpy=True, use_torch=True)
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
        load_models(CONFIG.LOAD_PATH, nfq_net=nfq_net, optimizer=optimizer)

    # NFQ Main loop
    # A set of transition samples denoted as D
    all_rollouts = []
    total_cost = 0
    if CONFIG.INIT_EXPERIENCE:
        for _ in range(CONFIG.INIT_EXPERIENCE):
            rollout, episode_cost = train_env.generate_rollout(
                None, render=CONFIG.TRAIN_RENDER
            )
            all_rollouts.extend(rollout)
            total_cost += episode_cost
    for epoch in range(CONFIG.EPOCH + 1):
        # Variant 1: Incermentally add transitions (Section 3.4)
        # TODO(seungjaeryanlee): Done before or after training?
        if CONFIG.INCREMENT_EXPERIENCE:
            new_rollout, episode_cost = train_env.generate_rollout(
                nfq_agent.get_best_action, render=CONFIG.TRAIN_RENDER
            )
            all_rollouts.extend(new_rollout)
            total_cost += episode_cost

        state_action_b, target_q_values = nfq_agent.generate_pattern_set(all_rollouts)

        # Variant 2: Clamp function to zero in goal region
        # TODO(seungjaeryanlee): Since this is a regulator setting, should it
        #                        not be clamped to zero?
        if CONFIG.HINT_TO_GOAL:
            goal_state_action_b, goal_target_q_values = train_env.get_goal_pattern_set()
            goal_state_action_b = torch.FloatTensor(goal_state_action_b)
            goal_target_q_values = torch.FloatTensor(goal_target_q_values)
            state_action_b = torch.cat([state_action_b, goal_state_action_b], dim=0)
            target_q_values = torch.cat([target_q_values, goal_target_q_values], dim=0)

        loss = nfq_agent.train((state_action_b, target_q_values))

        # TODO(seungjaeryanlee): Evaluation should be done with 3000 episodes
        eval_episode_length, eval_success, eval_episode_cost = nfq_agent.evaluate(
            eval_env, CONFIG.EVAL_RENDER
        )

        if CONFIG.INCREMENT_EXPERIENCE:
            logger.info(
                "Epoch {:4d} | Train {:3d} / {:4.2f} | Eval {:4d} / {:5.2f} | Train Loss {:.4f}".format(  # noqa: B950
                    epoch,
                    len(new_rollout),
                    episode_cost,
                    eval_episode_length,
                    eval_episode_cost,
                    loss,
                )
            )
            if CONFIG.USE_TENSORBOARD:
                writer.add_scalar("train/episode_length", len(new_rollout), epoch)
                writer.add_scalar("train/episode_cost", episode_cost, epoch)
                writer.add_scalar("train/loss", loss, epoch)
                writer.add_scalar("eval/episode_length", eval_episode_length, epoch)
                writer.add_scalar("eval/episode_cost", eval_episode_cost, epoch)
            if CONFIG.USE_WANDB:
                wandb.log({"Train Episode Length": len(new_rollout)}, step=epoch)
                wandb.log({"Train Episode Cost": episode_cost}, step=epoch)
                wandb.log({"Train Loss": loss}, step=epoch)
                wandb.log(
                    {"Evaluation Episode Length": eval_episode_length}, step=epoch
                )
                wandb.log({"Evaluation Episode Cost": eval_episode_cost}, step=epoch)
        else:
            logger.info(
                "Epoch {:4d} | Eval {:4d} / {:5.2f} | Train Loss {:.4f}".format(
                    epoch, eval_episode_length, eval_episode_cost, loss
                )
            )
            if CONFIG.USE_TENSORBOARD:
                writer.add_scalar("train/loss", loss, epoch)
                writer.add_scalar("eval/episode_length", eval_episode_length, epoch)
                writer.add_scalar("eval/episode_cost", eval_episode_cost, epoch)
            if CONFIG.USE_WANDB:
                wandb.log({"Train Loss": loss}, step=epoch)
                wandb.log(
                    {"Evaluation Episode Length": eval_episode_length}, step=epoch
                )
                wandb.log({"Evaluation Episode Cost": eval_episode_cost}, step=epoch)

        if eval_success:
            logger.info(
                "Epoch {:4d} | Total Cycles {:6d} | Total Cost {:4.2f}".format(
                    epoch, len(all_rollouts), total_cost
                )
            )
            if CONFIG.USE_TENSORBOARD:
                writer.add_scalar("summary/total_cycles", len(all_rollouts), epoch)
                writer.add_scalar("summary/total_cost", total_cost, epoch)
            if CONFIG.USE_WANDB:
                wandb.log({"Total Cycles": len(all_rollouts)}, step=epoch)
                wandb.log({"Total Cost": total_cost}, step=epoch)
            break

    # Save trained agent
    if CONFIG.SAVE_PATH:
        save_models(CONFIG.SAVE_PATH, nfq_net=nfq_net, optimizer=optimizer)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
