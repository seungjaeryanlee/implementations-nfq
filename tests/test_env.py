"""Stub unit tests."""
import numpy as np
import pytest

from environments import CartPoleRegulatorEnv as Env


class TestCartPoleRegulatorEnv:
    def test_train_mode_reset(self):
        """Test reset() in train mode."""
        train_env = Env(mode="train")
        x, x_, theta, theta_ = train_env.reset()

        assert abs(x) <= 2.3
        assert x_ == 0
        assert abs(theta) <= 0.3
        assert theta_ == 0

    def test_eval_mode_reset(self):
        """Test reset() in eval mode."""
        eval_env = Env(mode="eval")
        x, x_, theta, theta_ = eval_env.reset()

        assert abs(x) <= 1.0
        assert x_ == 0
        assert abs(theta) <= 0.3
        assert theta_ == 0

    @pytest.mark.parametrize("env", [Env(mode="train"), Env(mode="eval")])
    def test_get_goal_pattern_set(self, env):
        """Test get_goal_pattern_set()."""
        goal_state_action_b, goal_target_q_values = env.get_goal_pattern_set()

        for x, _, theta, _, action in goal_state_action_b:
            assert abs(x) <= env.x_success_range
            assert abs(theta) <= env.theta_success_range
            assert action in [0, 1]
        for target in goal_target_q_values:
            assert target == 0

    @pytest.mark.parametrize("env", [Env(mode="train"), Env(mode="eval")])
    @pytest.mark.parametrize("get_best_action", [None, lambda x: 0])
    def test_generate_rollout_next_obs(self, env, get_best_action):
        """Test generate_rollout() generates continued observation."""
        env = Env(mode="train")
        rollout, episode_cost = env.generate_rollout(get_best_action=None)

        prev_next_obs = rollout[0][3]
        for obs, _, _, next_obs, _ in rollout[1:]:
            assert np.array_equal(prev_next_obs, obs)
            prev_next_obs = next_obs

    @pytest.mark.parametrize("env", [Env(mode="train"), Env(mode="eval")])
    @pytest.mark.parametrize("get_best_action", [None, lambda x: 0])
    def test_generate_rollout_cost_threshold(self, env, get_best_action):
        """Test generate_rollout() does not have a cost over 1."""
        env = Env(mode="train")
        rollout, episode_cost = env.generate_rollout(get_best_action=None)

        for (_, _, cost, _, _) in rollout:
            assert 0 <= cost <= 1

    @pytest.mark.parametrize("env", [Env(mode="train"), Env(mode="eval")])
    @pytest.mark.parametrize("get_best_action", [None, lambda x: 0])
    def test_generate_rollout_episode_cost(self, env, get_best_action):
        """Test generate_rollout()'s second return value episode_cost."""
        env = Env(mode="train")
        rollout, episode_cost = env.generate_rollout(get_best_action=None)

        total_cost = 0
        for _, _, cost, _, _ in rollout:
            total_cost += cost
        assert episode_cost == total_cost

    @pytest.mark.parametrize("env", [Env(mode="train"), Env(mode="eval")])
    @pytest.mark.parametrize("get_best_action", [None, lambda x: 0])
    def test_generate_rollout_with_random_action_done_value(self, env, get_best_action):
        """Test done values of generate_rollout()e."""
        env = Env(mode="train")
        rollout, episode_cost = env.generate_rollout(get_best_action)

        for i, (_, _, _, _, done) in enumerate(rollout):
            if i + 1 < len(rollout):
                assert not done
            else:
                assert done or len(rollout) == env.max_steps

    @pytest.mark.parametrize("env", [Env(mode="train"), Env(mode="eval")])
    def test_generate_rollout_gest_best_action(self, env):
        """Test generate_rollout() uses get_best_action correctly."""
        env = Env(mode="train")
        rollout, _ = env.generate_rollout(get_best_action=lambda x: 0)

        for _, action, _, _, _ in rollout:
            assert action == 0
