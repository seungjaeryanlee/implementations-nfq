import gym

class RegulatorWrapper(gym.Wrapper):
    def __init__(self, env, c_trans=0.01):
        gym.Wrapper.__init__(self, env)
        self.c_trans = c_trans

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        if -0.05 < obs[0] < 0.05:
            reward = 0
        else:
            reward = self.c_trans

        return obs, reward, done, info


def make_cartpole():
    env = gym.make('CartPole-v0')
    env._max_episode_steps = 100
    # env.spec.tags['wrapper_config.TimeLimit.max_episode_steps'] = 100
    env = RegulatorWrapper(env)

    return env
