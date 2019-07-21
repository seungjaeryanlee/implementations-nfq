"""Environment to run Neural Fitted Q-Iteration on."""
from typing import Dict, List, Tuple

import gym


class RegulatorWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, c_trans: float = 0.01, max_steps: int = 100):
        """Wrap CartPole-v0 to CartPole regulator problem in NFQ paper.

        Parameters
        ----------
        env : gym.Env
            The CartPole environment to wrap.
        c_trans : float
            Positive constant cost for every step.
        max_steps : int
            Time limit for an episode.

        """
        gym.Wrapper.__init__(self, env)
        self.c_trans = c_trans
        self.step_counter = 0
        self.max_steps = max_steps

    def _reset(self) -> List:
        """Reset environment.

        Returns
        -------
        obs : List
            Initial observation of CartPole.

        """
        obs = self.env.reset()
        self.step_counter = 0

        return obs

    def _step(self, action: int) -> Tuple[List, float, bool, Dict]:
        """Take a step in the environment.

        Parameters
        ----------
        action : int
            Action to take on the environment.

        Returns
        -------
        next_obs : List
            The next observation after taking a step.
        rew : float
            Reward given on this timestep.
        done : bool
            Is True if the episode is over, otherwise false.
        info : dict
            Additional information from this step.

        """
        obs, _, done, info = self.env.step(action)
        self.step_counter += 1

        # Compute 'done'
        if self.step_counter >= self.max_steps:  # Max Time Steps
            done = True
        elif obs[0] < -2.4 or obs[0] > 2.4:  # Failure State
            done = True
        else:
            done = False

        # Compute 'cost' and 'success'
        if -0.05 < obs[0] < 0.05:  # Success State
            cost = 0
            success = True
        elif obs[0] < -2.4 or obs[0] > 2.4:  # Failure State
            cost = 1
            success = False
        else:
            cost = self.c_trans
            success = False

        info["success"] = success
        return obs, cost, done, info


def make_cartpole(max_steps: int = 100) -> gym.Env:
    """Initialize a CartPole regulator environment for NFQ.

    Parameters
    ----------
    max_steps : int
        Time limit for an episode.

    Returns
    -------
    env : gym.Env
        The CartPole regulator environment.

    """
    gym.logger.set_level(40)  # Disable warnings
    env = gym.make("CartPole-v0")
    env._max_episode_steps = max_steps
    env = RegulatorWrapper(env, max_steps=max_steps)

    return env
