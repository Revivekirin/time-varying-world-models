import os
import unittest
from tvwm.envs.process_env import WorkerTransport
from tvwm.envs.prefetch_env import PrefetchedEnvironment, prepared_worker, environment_class


def fake_imports(): return {'cpu_only':True}


def fake_activate(connection,task,seed,parent_pid):
    connection.send({'status':'ok','payload':dict(task=task,seed=seed)})
    command,_=connection.recv()
    assert command=='close'
    connection.send({'status':'ok','payload':{}})


class PrefetchTests(unittest.TestCase):
    def test_prepare_then_explicit_activation(self):
        worker=WorkerTransport(prepared_worker,(os.getpid(),fake_imports,fake_activate),timeout=5)
        try:
            self.assertEqual(worker.receive(),{'cpu_only':True})
            self.assertEqual(worker.request('activate',('PushCube-v1',1100004)),dict(task='PushCube-v1',seed=1100004))
            worker.close()
            self.assertFalse(worker.process.is_alive())
        finally: worker.abort()

    def test_finish_preserves_prepared_but_close_cancels_it(self):
        events=[]
        class Worker:
            def close(self): events.append('closed-active'); return {}
            def abort(self): events.append('cancelled-prepared')
        env=PrefetchedEnvironment('PushCube-v1'); env._worker=Worker(); env._prepared=Worker()
        env.finish_episode()
        self.assertEqual(events,['closed-active'])
        env.close(); self.assertEqual(events,['closed-active','cancelled-prepared'])

    def test_failed_active_close_also_cancels_prepared(self):
        events=[]
        class Active:
            def close(self): raise RuntimeError('close failed')
        class Prepared:
            def abort(self): events.append('cancelled')
        env=PrefetchedEnvironment('PushCube-v1'); env._worker=Active(); env._prepared=Prepared()
        with self.assertRaisesRegex(RuntimeError,'close failed'): env.close()
        self.assertEqual(events,['cancelled'])

    def test_pipeline_is_explicit(self):
        self.assertEqual(environment_class('baseline').lifecycle,'spawn_per_episode_v1')
        self.assertEqual(environment_class('cpu-prefetch').lifecycle,'spawn_cpu_prefetch_v1')
        with self.assertRaises(ValueError): environment_class('unknown')

class PrefetchRngTests(unittest.TestCase):
    def test_preparation_activation_preserve_current_rng_and_seed(self):
        try:
            import torch
            import numpy as np
        except ImportError:
            self.skipTest('requires server tensor dependencies')
        if not torch.cuda.is_available(): self.skipTest('requires CUDA server')
        import random
        from unittest.mock import patch
        from tvwm.agents.resume import rng_state
        activations=[]
        class Transport:
            class Process: pid=123
            process=Process()
            def __init__(self,*args):
                random.random(); np.random.random(); torch.rand(1); torch.rand(1,device='cuda')
            def receive(self): return {'cpu_import_seconds':0.,'peak_rss_bytes':0}
            def request(self,command,payload):
                activations.append(payload)
                torch.rand(1); torch.rand(1,device='cuda')
                return dict(obs=np.zeros(35,dtype=np.float32),contract={'action_dim':8,'horizon':50},goal_distance=1.,memory={})
            def close(self): return {}
            def abort(self): pass
        env=PrefetchedEnvironment('PushCube-v1')
        with patch('tvwm.envs.prefetch_env.WorkerTransport',Transport):
            for seed in (1100003,1100004):
                random.random(); np.random.random(); torch.rand(1); torch.rand(1,device='cuda')
                before=rng_state(); env.reset(seed); after=rng_state()
                self.assertEqual(before['python'],after['python'])
                self.assertEqual(before['numpy'][0],after['numpy'][0])
                np.testing.assert_array_equal(before['numpy'][1],after['numpy'][1])
                self.assertEqual(before['numpy'][2:],after['numpy'][2:])
                self.assertTrue(torch.equal(before['cpu'],after['cpu']))
                for left,right in zip(before['cuda'],after['cuda']): self.assertTrue(torch.equal(left,right))
                env.done=True; env.finish_episode()
            env.close()
        self.assertEqual(activations,[('PushCube-v1',1100003),('PushCube-v1',1100004)])
