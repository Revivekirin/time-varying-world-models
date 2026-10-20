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
