"""Real spawn/pipe failure tests, without a simulator or CUDA dependency."""
import os
import time
import unittest
from tvwm.envs.process_env import WorkerTransport, WorkerError, EpisodeEnvironment


def fake_worker(connection, mode):
    if mode=='error':
        connection.send({'status':'error','error':'simulated failure','traceback':'child traceback'})
        connection.close(); return
    connection.send({'status':'ok','payload':os.getpid()})
    command,_=connection.recv()
    if mode=='exit': os._exit(17)
    if mode=='hang': time.sleep(60)
    if mode=='close_exit': os._exit(19)
    connection.send({'status':'ok','payload':command})
    connection.close()


class ProcessTests(unittest.TestCase):
    def test_spawn_and_close(self):
        worker=WorkerTransport(fake_worker,('ok',),timeout=5)
        self.assertNotEqual(worker.receive(),os.getpid())
        self.assertEqual(worker.close(),'close')
        self.assertFalse(worker.process.is_alive())
        self.assertTrue(worker.connection.closed)

    def test_native_exit(self):
        worker=WorkerTransport(fake_worker,('exit',),timeout=5)
        self.assertNotEqual(worker.receive(),os.getpid())
        with self.assertRaisesRegex(WorkerError,'exit'): worker.request('exit')
        self.assertFalse(worker.process.is_alive())
        self.assertTrue(worker.connection.closed)

    def test_child_exception(self):
        worker=WorkerTransport(fake_worker,('error',),timeout=5)
        with self.assertRaisesRegex(WorkerError,'child traceback'): worker.receive()
        self.assertFalse(worker.process.is_alive())

    def test_timeout_cleanup(self):
        worker=WorkerTransport(fake_worker,('hang',),timeout=5)
        self.assertNotEqual(worker.receive(),os.getpid())
        worker.timeout=.3
        with self.assertRaisesRegex(WorkerError,'timeout'): worker.request('hang')
        self.assertFalse(worker.process.is_alive())
        self.assertTrue(worker.connection.closed)

    def test_shutdown_native_exit(self):
        worker=WorkerTransport(fake_worker,('close_exit',),timeout=5)
        worker.receive()
        with self.assertRaises(WorkerError): worker.close()
        self.assertFalse(worker.process.is_alive())

    def test_incomplete_episode_cannot_commit(self):
        env=EpisodeEnvironment('PushCube-v1'); env.done=False
        with self.assertRaisesRegex(RuntimeError,'incomplete'): env.finish_episode()

class ParentRngTests(unittest.TestCase):
    def test_reset_preserves_all_current_rng_streams(self):
        try:
            import torch
            import numpy as np
        except ImportError:
            self.skipTest('requires server tensor dependencies')
        from unittest.mock import patch
        from tvwm.agents.resume import rng_state
        import random
        class FakeTransport:
            class Process: pid=42
            process=Process()
            def __init__(self,*args):
                random.random(); np.random.random(); torch.rand(3)
                if torch.cuda.is_available(): torch.rand(3,device='cuda')
            def receive(self):
                return dict(obs=np.zeros(35,dtype=np.float32),contract={'action_dim':8,'horizon':50},goal_distance=1.,memory={})
            def close(self): return {}
        if not torch.cuda.is_available(): self.skipTest('requires CUDA server')
        env=EpisodeEnvironment('PushCube-v1'); before=rng_state()
        with patch('tvwm.envs.process_env.WorkerTransport',FakeTransport): env.reset(1100000)
        after=rng_state(); env.close()
        self.assertEqual(before['python'],after['python'])
        self.assertEqual(before['numpy'][0],after['numpy'][0])
        np.testing.assert_array_equal(before['numpy'][1],after['numpy'][1])
        self.assertEqual(before['numpy'][2:],after['numpy'][2:])
        self.assertTrue(torch.equal(before['cpu'],after['cpu']))
        self.assertEqual(len(before['cuda']),len(after['cuda']))
        for left,right in zip(before['cuda'],after['cuda']): self.assertTrue(torch.equal(left,right))
