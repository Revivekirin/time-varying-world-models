"""Verify actual shared collection ordering at the upstream warmup discontinuity."""
import unittest
from unittest.mock import patch
try:
    import torch
    from tvwm.agents.nominal import collect, append_episode
except ImportError:
    torch=None


@unittest.skipIf(torch is None, 'requires server torch dependencies')
class CollectorTests(unittest.TestCase):
    def run_episode(self, start, length, limit=300000, close_failure=False):
        events=[]
        class Env:
            action_dim=8
            def reset(self, seed):
                self.t=0; return torch.zeros(35),{}
            def close(self): events.append('discard-close')
            def finish_episode(self):
                events.append('close')
                if close_failure: raise RuntimeError('native worker close failure')
            def step(self, action):
                self.t+=1; events.append('step')
                return torch.zeros(35),torch.tensor(1.),False,self.t==length,{'success':torch.tensor(False)}
        class Agent:
            def act(self, obs, **kw):
                events.append('act'); return torch.zeros(8)
            def update(self, replay):
                events.append('update'); return {'total_loss':1.}
        class Replay:
            def add(self, td): events.append('add')
        counters=dict(transitions=start,updates=0,episode_index=0)
        with patch('tvwm.agents.nominal.goal_distance',return_value=0.),patch('torch.cuda.synchronize'):
            result=collect(Env(),Agent(),Replay(),counters,1,limit=limit)
        return events,counters,result

    def test_seed_boundary_burst_then_policy(self):
        events,c,r=self.run_episode(1000,2)
        self.assertEqual(c['updates'],1001)
        self.assertEqual(events[:2],['step','update'])
        self.assertEqual(events.count('act'),1)
        self.assertEqual(events[-1],'add')
        self.assertEqual(c['episode_index'],1)

    def test_partial_budget_does_not_enter_replay(self):
        events,c,r=self.run_episode(299999,50)
        self.assertEqual(c['transitions'],300000)
        self.assertEqual(c['updates'],1)
        self.assertNotIn('add',events)
        self.assertFalse(r['complete'])
        self.assertEqual(c['episode_index'],0)

    def test_worker_close_failure_never_inserts_replay(self):
        with patch('tvwm.agents.nominal.append_episode') as insert:
            with self.assertRaisesRegex(RuntimeError,'native worker close failure'):
                self.run_episode(0,1,close_failure=True)
            insert.assert_not_called()

    def test_random_episode_no_updates(self):
        events,c,r=self.run_episode(0,50)
        self.assertEqual(events.count('step'),50)
        self.assertEqual(events.count('update'),0)
        self.assertEqual(events.count('act'),0)
        self.assertEqual(events[-1],'add')


@unittest.skipIf(torch is None, 'requires server torch dependencies')
class RestoredReplayTests(unittest.TestCase):
    def check_device(self, device):
        import io
        from tensordict import TensorDict
        from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
        original=ReplayBuffer(storage=LazyTensorStorage(32,device=device),batch_size=2)
        original.extend(TensorDict({'obs':torch.ones(4,3), 'reward':torch.ones(4)},[4]).to(device))
        serialized=io.BytesIO()
        torch.save(original.state_dict(),serialized); serialized.seek(0)
        restored=ReplayBuffer(storage=LazyTensorStorage(32,device=device),batch_size=2)
        restored.load_state_dict(torch.load(serialized,weights_only=False))
        owner=self
        class Adapter:
            _storage_device=torch.device(device)
            def add(self, episode):
                owner.assertEqual(episode.device, self._storage_device)
                return restored.extend(episode)
        incoming=TensorDict({'obs':torch.full((3,3),2.), 'reward':torch.full((3,),2.)},[3])
        append_episode(Adapter(),incoming)
        self.assertEqual(len(restored),7)
        torch.testing.assert_close(restored._storage[4:7]['obs'].cpu(),torch.full((3,3),2.))
        self.assertEqual(incoming['obs'].device.type,'cpu')

    def test_restored_cpu_storage_append(self):
        self.check_device('cpu')

    def test_restored_cuda_storage_append(self):
        if not torch.cuda.is_available(): self.skipTest('requires CUDA server')
        self.check_device('cuda:0')
