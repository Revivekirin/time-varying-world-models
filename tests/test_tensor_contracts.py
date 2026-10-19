import importlib.util
import unittest

@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('tensordict'), 'requires reviewed tensor environment')
class TensorContractTests(unittest.TestCase):
    def test_observation_conversion(self):
        import torch
        from tvwm.envs.maniskill import observation
        x=torch.tensor([[1.,2.]],dtype=torch.float64)
        y=observation(x)
        self.assertEqual(y.dtype,torch.float32)
        self.assertEqual(tuple(y.shape),(2,))
        with self.assertRaises(ValueError): observation(x.repeat(2,1))
        with self.assertRaises(ValueError): observation(torch.tensor([[float('nan')]]))
    def test_episode_tensor(self):
        import torch
        from tvwm.agents.upstream import episode_tensor
        td=episode_tensor([torch.tensor([1.]),torch.tensor([2.])],[torch.tensor([.3])],[4.],[False],[True])
        self.assertTrue(torch.isnan(td['action'][0]).all())
        self.assertEqual(td['obs'][:,0].tolist(),[1,2])
        self.assertEqual(td['reward'][1].item(),4)
        self.assertEqual(td['terminated'][1].item(),0)
