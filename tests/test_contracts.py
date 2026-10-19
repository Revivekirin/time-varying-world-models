import unittest
from tvwm.envs.contracts import affine_action, sequence_rows

class ContractTests(unittest.TestCase):
    def test_asymmetric_action_roundtrip(self):
        low=[-0.1,-0.01];high=[0.1,0.04]
        for a in ([-1,1],[0,0],[0.5,-0.25]):
            native=affine_action(a,low,high)
            restored=affine_action(native,low,high,inverse=True)
            for actual,expected in zip(restored,a): self.assertAlmostEqual(actual,expected)
    def test_invalid_action(self):
        for a,l,h in [([2],[-1],[1]),([float('nan')],[-1],[1]),([0],[1],[1]),([0,1],[-1],[1])]:
            with self.assertRaises(ValueError): affine_action(a,l,h)
    def test_replay_alignment(self):
        rows=sequence_rows(['o0','o1','o2'],['a0','a1'],[2,7],[False,True],[False,False])
        self.assertEqual(rows,[('o0',None,None,None),('o1','a0',2,False),('o2','a1',7,True)])
    def test_truncation_bootstraps(self):
        rows=sequence_rows([0,1],[9],[3],[False],[True])
        self.assertFalse(rows[-1][-1])
    def test_terminal_dominates_simultaneous_flags(self):
        self.assertTrue(sequence_rows([0,1],[9],[3],[True],[True])[-1][-1])
    def test_cross_episode_rejected(self):
        with self.assertRaises(ValueError): sequence_rows([0,1,2],[1,2],[3,4],[True,False],[False,True])
    def test_incomplete_rejected(self):
        with self.assertRaises(ValueError): sequence_rows([0,1],[1],[2],[False],[False])
    def test_lengths_rejected(self):
        with self.assertRaises(ValueError): sequence_rows([0],[1],[2],[True],[False])

if __name__=='__main__': unittest.main()
