import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('root_cause',Path(__file__).resolve().parents[1]/'scripts/phase1_root_cause.py')
diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)
try:
    import torch
except ImportError:
    torch=None

class FlattenTests(unittest.TestCase):
    def test_nested_rng_paths(self):
        self.assertEqual(diag.flatten({'cuda':[1,2],'cpu':3}),{'/cuda/0':1,'/cuda/1':2,'/cpu':3})

@unittest.skipIf(torch is None,'requires torch')
class DifferenceTests(unittest.TestCase):
    def test_exact_tolerance_and_failure_are_separate(self):
        result=diag.compare({'x':torch.tensor([0.,0.,0.])},{'x':torch.tensor([0.,2.98e-8,4.49e-5])})
        self.assertFalse(result['exact']);self.assertFalse(result['within_tolerance'])
        self.assertEqual(result['fields']['x']['first_nonzero_index'],[1])
        self.assertEqual(result['fields']['x']['first_outside_tolerance_index'],[2])
    def test_scalar_rng_and_missing_state(self):
        self.assertFalse(diag.compare({'a':torch.tensor(1)},{'a':torch.tensor(2)})['exact'])
        self.assertFalse(diag.compare({'a':torch.tensor(1)},{})['within_tolerance'])


@unittest.skipIf(torch is None,'requires torch')
class StateDigestTests(unittest.TestCase):
    def test_metadata_and_tensor_changes(self):
        state={'weight':torch.tensor(1.),'shape':torch.Size([2,3]),
               'device':torch.device('cpu'),'nested':{'dtype':torch.float32,'value':None}}
        cloned={'nested':{'value':None,'dtype':torch.float32},'device':torch.device('cpu'),
                'shape':torch.Size([2,3]),'weight':torch.tensor(1.)}
        self.assertEqual(diag.state_digest(state),diag.state_digest(cloned))
        cloned['shape']=torch.Size([3,2])
        self.assertNotEqual(diag.state_digest(state),diag.state_digest(cloned))
        cloned['shape']=state['shape'];cloned['weight']=torch.tensor(2.)
        self.assertNotEqual(diag.state_digest(state),diag.state_digest(cloned))

    def test_real_tensordict_state(self):
        from tensordict import TensorDict
        state=TensorDict({'w':torch.ones(2)},batch_size=[]).state_dict()
        self.assertEqual(diag.state_digest(state),diag.state_digest(state))

    def test_bfloat_and_empty_tensor(self):
        self.assertEqual(diag.state_digest({'x':torch.ones(2,dtype=torch.bfloat16),'empty':torch.empty(0)}),
                         diag.state_digest({'empty':torch.empty(0),'x':torch.ones(2,dtype=torch.bfloat16)}))
