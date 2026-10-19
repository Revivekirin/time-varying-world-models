"""Time-limit discovery without constructing a simulator."""
import importlib.util
import unittest

@unittest.skipUnless(importlib.util.find_spec('mani_skill') and importlib.util.find_spec('gymnasium'),
                     'requires installed ManiSkill/Gymnasium')
class HorizonTests(unittest.TestCase):
    def test_wrapper_limit_when_spec_has_no_limit(self):
        import gymnasium as gym
        from gymnasium.envs.registration import EnvSpec
        from mani_skill.utils.gym_utils import find_max_episode_steps_value
        base = gym.Env()
        base.spec = EnvSpec('ContractFixture-v0', max_episode_steps=None)
        wrapper = gym.Wrapper(base)
        wrapper._max_episode_steps = 50
        self.assertIsNone(wrapper.spec.max_episode_steps)
        self.assertEqual(find_max_episode_steps_value(wrapper), 50)
        wrapper._max_episode_steps = 33
        self.assertEqual(find_max_episode_steps_value(wrapper), 33)
