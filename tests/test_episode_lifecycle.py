import random
import unittest
import weakref
from contextlib import contextmanager
from unittest.mock import patch
from tvwm.envs.episode_lifecycle import EpisodeEnvironment


@contextmanager
def fake_rng(seed):
    current=random.getstate()
    try:
        random.seed(seed)
        yield
    finally:
        random.setstate(current)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.events=[]; self.refs=[]; self.finish=True; self.drift=False
        owner=self
        class Adapter:
            def __init__(self,task):
                owner.events.append('construct'); self.done=True
                random.random()  # Simulate constructor consumption.
                self.contract={'task':task,'action_dim':9 if owner.drift else 8}
                owner.refs.append(weakref.ref(self))
            def reset(self,seed):
                owner.events.append(('reset',seed)); self.done=False
                return seed,{}
            def step(self,action):
                self.done=owner.finish
                return action,0.,False,self.done,{}
            def close(self):
                owner.events.append('close'); random.random()
        self.factory=patch('tvwm.envs.episode_lifecycle.adapter_factory',Adapter)
        self.rng=patch('tvwm.envs.episode_lifecycle.construction_rng',fake_rng)
        self.sync=patch('tvwm.envs.episode_lifecycle.synchronize')
        for p in (self.factory,self.rng,self.sync): p.start();self.addCleanup(p.stop)

    def test_recreate_release_and_seed(self):
        env=EpisodeEnvironment('PushCube-v1');env.reset(1100000);env.step(0)
        env.reset(1100001)
        self.assertEqual(self.events,['construct',('reset',1100000),'close','construct',('reset',1100001)])
        self.assertIsNone(self.refs[0]())
        self.assertEqual(env.recreation_count,2)
        env.close();env.close()
        self.assertIsNone(self.refs[1]())

    def test_no_mid_episode_recreation(self):
        env=EpisodeEnvironment('PushCube-v1');env.reset(1);self.finish=False;env.step(0)
        with self.assertRaisesRegex(RuntimeError,'midway'):env.reset(2)
        self.assertEqual(env.recreation_count,1);env.close()

    def test_preserves_current_rng_not_previous_checkpoint_stream(self):
        env=EpisodeEnvironment('PushCube-v1');random.seed(7);before=random.getstate()
        env.reset(1);self.assertEqual(random.getstate(),before)
        random.random();random.random();after_training=random.getstate()
        env.step(0);env.reset(2)
        self.assertEqual(random.getstate(),after_training);env.close()

    def test_contract_drift_fails_and_releases(self):
        env=EpisodeEnvironment('PushCube-v1');env.reset(1);env.step(0);self.drift=True
        with self.assertRaisesRegex(ValueError,'contract mismatch'):env.reset(2)
        self.assertIsNone(env._adapter)

    def test_constructor_failure_restores_rng(self):
        env=EpisodeEnvironment('PushCube-v1');random.seed(3);before=random.getstate()
        def fail(task):
            random.random();raise RuntimeError('construction failed')
        with patch('tvwm.envs.episode_lifecycle.adapter_factory',fail):
            with self.assertRaisesRegex(RuntimeError,'construction failed'):env.reset(2)
        self.assertEqual(random.getstate(),before);self.assertIsNone(env._adapter)


class RealRngTests(unittest.TestCase):
    def test_python_numpy_cpu_and_cuda_streams(self):
        try:
            import numpy as np
            import torch
            from tvwm.agents.resume import rng_state
            from tvwm.envs.episode_lifecycle import construction_rng
        except ImportError:
            self.skipTest('requires server tensor dependencies')
        before=rng_state()
        with construction_rng(123):
            random.random();np.random.rand();torch.rand(5)
            if torch.cuda.is_available():torch.rand(5,device='cuda')
        after=rng_state()
        self.assertEqual(before['python'],after['python'])
        self.assertEqual(before['numpy'][0],after['numpy'][0])
        np.testing.assert_array_equal(before['numpy'][1],after['numpy'][1])
        self.assertEqual(before['numpy'][2:],after['numpy'][2:])
        self.assertTrue(torch.equal(before['cpu'],after['cpu']))
        self.assertEqual(len(before['cuda']),len(after['cuda']))
        for left,right in zip(before['cuda'],after['cuda']):self.assertTrue(torch.equal(left,right))
