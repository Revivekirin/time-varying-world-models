import unittest
from tvwm.agents.cadence import updates_at, random_action_at

class CadenceTests(unittest.TestCase):
    def test_upstream_boundary(self):
        self.assertEqual([updates_at(s) for s in [999,1000,1001]],[0,1000,1])
        self.assertEqual([random_action_at(s) for s in [999,1000,1001]],[True,True,False])
    def test_exact_budget(self):
        self.assertEqual(sum(updates_at(s) for s in range(300000)),299999)
        self.assertEqual(sum(random_action_at(s) for s in range(300000)),1001)
    def test_invalid_steps(self):
        with self.assertRaises(ValueError):updates_at(-1)
