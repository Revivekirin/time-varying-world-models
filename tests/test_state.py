import importlib.util
import unittest
from tvwm.utils.state import snapshot_state, assert_state_equal

@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires torch')
class StateTests(unittest.TestCase):
    def test_mixed_state_snapshot_and_comparison(self):
        import torch
        from collections import OrderedDict
        state=OrderedDict(weight=torch.tensor([1.]), batch_size=torch.Size([5]), device=torch.device('cpu'), optional=None)
        state._metadata={'': {'version':1}}
        snapshot=snapshot_state(state)
        assert_state_equal(state,snapshot)
        self.assertEqual(snapshot._metadata,state._metadata)
        state['weight'].add_(1)
        self.assertEqual(snapshot['weight'].item(),1)
        with self.assertRaises(AssertionError): assert_state_equal(state,snapshot)
        state['weight'].sub_(1)
        state['batch_size']=torch.Size([6])
        with self.assertRaises(AssertionError): assert_state_equal(state,snapshot)

    @unittest.skipUnless(importlib.util.find_spec('tensordict'), 'requires tensordict')
    def test_real_tensordict_state(self):
        import torch
        from tensordict import TensorDict
        from tensordict.nn import TensorDictParams
        module=TensorDictParams(TensorDict({'weight':torch.ones(2,3)}, batch_size=[2]))
        state=module.state_dict()
        snapshot=snapshot_state(state)
        self.assertTrue(any(not isinstance(v,torch.Tensor) for v in state.values()))
        module.load_state_dict(snapshot)
        assert_state_equal(module.state_dict(),snapshot)
