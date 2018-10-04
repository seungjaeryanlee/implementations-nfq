import gym

class RegulatorWrapper(gym.Wrapper):
    def __init__(self, env, c_trans=0.01):
        gym.Wrapper.__init__(self, env)
        self.c_trans = c_trans

    def _step(self, action):
        obs, _, done, info = self.env.step(action)
        if -0.05 < obs[0] < 0.05:
            cost = 0
            success = True
        else:
            cost = self.c_trans
            success = False
        info['success'] = success

        return obs, cost, done, info


def make_cartpole(max_steps=100):
    env = gym.make('CartPole-v0')
    env._max_episode_steps = max_steps
    env = RegulatorWrapper(env)

    return env
