# Phase 1 nominal TD-MPC2 configuration — revised proposal

This supersedes the Phase1 pilot proposal in the original Phase0 research plan.
That Phase0 artifact is retained unchanged. There will be NO separate10000-step
training pilot. Phase1 aims at competent nominal policies, not a loss smoke pass.
No shifts/history conditioning/online adaptation, nor original TD-MPC baseline.
Remote execution remains exclusively user-operated.

| Setting | Proposed value |
|---|---|
| Tasks / training seed | PushCube-v1 and PickCube-v1; seed1 each initially |
| Interaction budget | exactly300000 transitions per task,600000 combined |
| Model / batch / envs | official TD-MPC2 model_size5 /256 /1 |
| Observation / actions | state35 or42, normalized8; Panda pd_joint_delta_pos |
| Reward / horizon / repeat | native normalized_dense /50 /1 |
| Episode support | episodic=true; terminated masks bootstrap, truncated closes episodes |
| Replay | upstream Buffer,300000 rows including initial observation rows |
| Planner | horizon3, iterations6,512 samples,64 elites,24 policy trajectories |
| Objectives / optimizer | unchanged pinned upstream defaults; discount.95 |
| Compile | false as in verified Phase0; performance choice, no equation change |
| Validation / checkpoint cadence | every10000 transitions, first completed episode at/after due point |
| Full-run deadline | proposed24h/task, includes evaluation/checkpoint overhead; cumulative across resumes |
| GPU / disk | proposed20GiB device-wide ceiling,4GiB CUDA headroom;30GiB free storage |

Machine-readable proposal: configs/phase1/nominal.json. Not launch approval.

## VERIFIED upstream cadence, inspected source

external/tdmpc2/tdmpc2/trainer/online_trainer.py uses zero-based _step.
Random action while _step<=seed_steps; updates begin after collecting the action
at _step==seed_steps, with seed_steps optimizer updates, then1 per later step.
For seed_steps1000 this means1001 random transitions, not1000. Original inclusive
while _step<=cfg.steps performs300001 transitions when cfg.steps300000 and300000
updates. Proposed external runner caps collection at300000 (indices0..299999),
preserving those action/update conditions:1000 burst +298999 later updates =
299999 learner updates per task. This endpoint change is explicit, not a silent
cadence correction. src/tvwm/agents/cadence.py has locally passing boundary tests.

Completed episodes enter replay at the subsequent reset boundary upstream.
Do not train on a reset observation as if it were the terminal next observation.
Maintain initial NaN sentinel row and native termination/truncation distinction.
Do not replace termination with success assumptions or force fixed50-step episodes.
Short completed episodes must not silently be extended or relabeled; SliceSampler
length handling and early-success prevalence need reporting in full training.

## Evaluation/checkpoint policy

Independent validation seeds11001..11020, disjoint from Phase0 seeds10001/10002.
Evaluate initial policy and at30 checkpoints, at most31000 validation transitions
per task. Final heldout seeds21001..21100 on validation-selected checkpoint only,
at most5000 transitions/task. Evaluation does not increment training counters or
modify replay, optimizer, policy RNG, or training environment state. Use a
separate process, eval_mode=True, model.eval and no gradients; planner sampling
remains seeded stochastic. Serial execution is preferred on the3090.

Log success-ever, terminal success, native return, episode length, term/trunc,
training/evaluation transition counts and episode seed. Goal progress is a metric,
not reward shaping: Push native cube-goal xy distance, Pick3D cube-goal distance;
record initial/final/minimum distance, and Pick placement/static criteria.
Keep metrics out of policy inputs. Select highest validation success fraction,
then mean return, then earliest checkpoint. Never use heldout or future shift
results for selection. Latest, best and final remain separate artifacts.

Full-run readiness still requires batch256 results, recorded installed lock,
collector/fresh-process resume tests, reviewed resource estimates and explicit
approval of the final300000/task budget. No executable full-training launcher
is delivered as ready in this revision; the runnable command is preflight only.
