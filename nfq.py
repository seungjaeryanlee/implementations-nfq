#!/usr/bin/env python
import random
import torch
import torch.optim as optim
import torch.nn.functional as F

import models
import envs


def set_random_seeds(train_env, test_env, seed=0xc0ffee):
    """
    Set random seeds for reproducibility.
    """
    train_env.seed(seed)
    test_env.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)

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

def generate_rollout(env, net=None):
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
    q_next_state_left_batch = net(torch.cat([next_state_batch, torch.zeros(len(rollout), 1)], 1)).squeeze()
    q_next_state_right_batch = net(torch.cat([next_state_batch, torch.ones(len(rollout), 1)], 1)).squeeze()
    q_next_state_batch = torch.min(q_next_state_left_batch, q_next_state_right_batch)

    with torch.no_grad():
        target_q_values = cost_batch + gamma * q_next_state_batch * (torch.FloatTensor(1) - done_batch)

    loss = F.mse_loss(predicted_q_values, target_q_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def hint_to_goal(net, optimizer, factor=100):
    """
    Use hint-to-goal heuristic to clamp 
    """
    for _ in range(factor):
        state_action_pair = torch.FloatTensor([[random.random() * 0.025 - 0.5,
                                                random.random(),
                                                random.random() * 0.3 - 0.6,
                                                random.random(),
                                                random.randint(0, 1)]])
        predicted_q_value = net(state_action_pair).squeeze()
        loss = F.mse_loss(predicted_q_value, torch.zeros(1))

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
            obs, cost, done, info = env.step(action)
            steps += 1

        nb_success += 1 if info['success'] else 0

    print('Average Number of Steps: ', float(steps) / episodes)
    print('Success rate: ', float(nb_success) / episodes)
    env.close()

def main():
    train_env = envs.make_cartpole(100)
    test_env = envs.make_cartpole(3000)
    # set_random_seeds(train_env, test_env)

    net = models.Net()
    optimizer = optim.Rprop(net.parameters())
    # TODO Initialize weights randomly within [-0.5, 0.5]

    for epoch in range(500):
        rollout = generate_rollout(train_env, net)
        if epoch % 10 == 9:
            print('Epoch {:4d} | Steps: {:3d}'.format(epoch + 1, len(rollout)))
        train(net, optimizer, rollout)
        hint_to_goal(net, optimizer)

    test(test_env, net)


if __name__ == '__main__':
    main()
