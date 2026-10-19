"""Verify actual shared collection ordering at the upstream warmup discontinuity."""
import unittest
from unittest.mock import patch
try:
    import torch
    from tvwm.agents.nominal import collect
except ImportError:
    torch=None


@unittest.skipIf(torch is None, 'requires server torch dependencies')
class CollectorTests(unittest.TestCase):
    def run_episode(self, start, length, limit=300000):
        events=[]
        class Env:
            action_dim=8
            def reset(self, seed):
                self.t=0; return torch.zeros(35),{}
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

    def test_random_episode_no_updates(self):
        events,c,r=self.run_episode(0,50)
        self.assertEqual(events.count('step'),50)
        self.assertEqual(events.count('update'),0)
        self.assertEqual(events.count('act'),0)
        self.assertEqual(events[-1],'add')
