import unittest
from tvwm.envs.contracts import compare_recorded_contract

class RecordedContractTests(unittest.TestCase):
    def test_tuple_list_roundtrip(self):
        compare_recorded_contract({'action_mapping':{'arm':(0,7),'gripper':(7,8)}},
                                  {'action_mapping':{'arm':[0,7],'gripper':[7,8]}}, ['action_mapping'])
    def test_changed_indices_rejected(self):
        with self.assertRaisesRegex(ValueError,'action_mapping'):
            compare_recorded_contract({'action_mapping':{'arm':(0,6)}},
                                      {'action_mapping':{'arm':[0,7]}}, ['action_mapping'])
    def test_scalar_change_rejected(self):
        with self.assertRaisesRegex(ValueError,'horizon'):
            compare_recorded_contract({'horizon':100},{'horizon':50},['horizon'])
    def test_missing_field_rejected(self):
        with self.assertRaises(KeyError): compare_recorded_contract({}, {'horizon':50}, ['horizon'])
