import gym

class RegulatorWrapper(gym.Wrapper):
    def __init__(self, env, c_trans=0.01, max_steps=100):
        """
        Wrap CartPole-v0 to CartPole regulator problem in NFQ paper.
        """
        gym.Wrapper.__init__(self, env)
        self.c_trans = c_trans
        self.step_counter = 0
        self.max_steps = max_steps

    def _reset(self):
        obs = self.env.reset()
        self.step_counter = 0

        return obs

    def _step(self, action):

        obs, _, done, info = self.env.step(action)
        self.step_counter += 1

        # Compute 'done'
        if self.step_counter >= self.max_steps: # Max Time Steps
            done = True
        elif obs[0] < -2.4 or obs[0] > 2.4: # Failure State
            done = True
        else:
            done = False

        # Compute 'cost' and 'success'
        if -0.05 < obs[0] < 0.05: # Success State
            cost = 0
            success = True
        elif obs[0] < -2.4 or obs[0] > 2.4: # Failure State
            cost = 1
            success = False
        else:
            cost = self.c_trans
            success = False

        info['success'] = success
        return obs, cost, done, info


def make_cartpole(max_steps=100):
    gym.logger.set_level(40) # Disable warnings
    env = gym.make('CartPole-v0')
    env._max_episode_steps = max_steps
    env = RegulatorWrapper(env, max_steps=max_steps)

    return env
