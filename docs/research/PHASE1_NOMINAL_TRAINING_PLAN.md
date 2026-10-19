# Phase 1 nominal training proposals — NOT AUTHORIZED TO RUN

Both tasks use official TD-MPC2 preset 5M, state, Panda pd_joint_delta_pos,
native normalized_dense reward, native horizon50, action_repeat1, episodic=true.
Keep upstream planner horizon3, samples512/elites64/pi24, iterations6; batch256,
Adam3e-4, tau.01, rho.5, coefficients20/.1/.1/1, discount derived to .95.
No controller/task/reward tuning or history features.

| Budget per task and seed | Short pilot proposal | Full proposal (separate approval) |
|---|---:|---:|
| Training env count | 1 | 1 |
| Interaction transitions | 10,000 | 1,000,000 |
| Seed random transitions | 1,000 | 1,000 |
| Learner updates | 10,000 total | 1,000,000 total |
| Initial update burst | exactly 1,000 after seed collection | exactly 1,000 after seed collection |
| Remaining ratio | 1 update / transition | 1 update / transition |
| Replay capacity | 10,000 rows + episode-start allowance | 1,000,000 rows |
| Training seeds | 1 | 1,2,3 |
| Checkpoint/eval cadence | every 2,000 steps plus final | every 50,000 steps plus final |

Budgets count transitions, not physics substeps or number of vectorized calls.
Replay includes one initial observation row per episode; row capacity is distinct
from transitions. Stop at exact budgets, insert final partial episode only under
a reviewed explicitly truncated collector contract; do not invent termination.
Nominal training runner remains to be implemented/reviewed after Phase 0 results;
smoke runner cannot launch these budgets. Unfinished episodes at a budget boundary
need an explicit data-accounting decision before full run.

Evaluate in a separate process with model frozen, no training reset interference.
Validation seeds10001–10020, final held-out20001–20100, same seeds across comparison
models. Log per-episode undiscounted native return, success-ever and terminal
success, length and flags; aggregate across evaluation and training seeds with
uncertainty. Choose highest validation success checkpoint, mean return as tie-break,
earliest checkpoint if still tied. Evaluate selected checkpoint once on held-out
set. Never choose checkpoint on final held-out performance. Pilot is for diagnosing
learning/resource trends, not satisfying competence by loss decrease.

GPU/time estimates UNRESOLVED until smoke measurements. Runtime approximation:
T ≈ transitions*(native_step_seconds + planner_seconds after seed) +
updates*update_seconds + evaluation overhead. Use synchronized measured rates,
report warmup separately, scale replay storage from tensor dimensions, and include
SAPIEN memory from device-wide nvidia-smi. A 16-batch smoke does not directly prove
256-batch memory fitness; propose a bounded batch256 capacity check for review
before pilot. 24GB target is a constraint, not a measured safety margin.

Stop for NaNs/infinite gradients, invalid actions/observations, OOM, replay boundary
mismatch, unexpected termination semantics, missing assets, or checkpoint mismatch.
No automatic tuning/restarts or budget expansion. Interruptions do not resume from
model-only snapshots. Competence criteria must be reviewed from task performance;
no unsupported success-rate cutoff is invented. Full run approval follows pilot
review; Phase 2 requires a suitable nominal checkpoint for frozen evaluation.
