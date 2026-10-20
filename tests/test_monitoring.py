import unittest
from tvwm.utils.monitoring import scalar_metrics

class MetricSchemaTests(unittest.TestCase):
    def test_distinct_step_axes(self):
        counters={'transitions':1001,'updates':1000,'evaluation_transitions':100}
        record=scalar_metrics('update',{'transitions':1001,'updates':1000,'seconds':.04,'total_loss':.2},counters)
        self.assertEqual(record['step/env'],1001)
        self.assertEqual(record['step/update'],1000)
        self.assertEqual(record['loss/total_loss'],.2)
        self.assertEqual(record['perf/learner_updates_per_s'],25.)
        self.assertNotIn('loss/updates',record)

    def test_episode_and_eval_names(self):
        counters={'transitions':10000,'updates':9999}
        record=scalar_metrics('episode',{'return_':5.,'success':False,'length':50,'planner_seconds':2.},counters)
        self.assertEqual(record['train/episode_return'],5.)
        self.assertEqual(record['perf/planner_ms_per_transition'],40.)
        evaluation=scalar_metrics('validation',{'success_rate':.5,'mean_return':10.,'return_std':2.,'goal_progress_mean':.1},counters)
        self.assertEqual(evaluation['eval/success_rate'],.5)
        self.assertEqual(evaluation['eval/return_std'],2.)
        self.assertEqual(evaluation['step/eval_env'],0)
