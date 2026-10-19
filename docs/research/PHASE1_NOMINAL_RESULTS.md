# Phase 1 nominal results

UNRESOLVED — no Phase1 full-training run executed. No nominal success or stable
control claim. Initial training seed count is1/task: evaluation-seed stability
can be measured, but training-seed reproducibility cannot be established with1.

VERIFIED user-provided Phase0 evidence: both environments' contract tests passed,
observations35/42, action8, native horizon50 via ManiSkill helper,100/20Hz;
GPU model/planner/replay,8-update learner, Polyak and model checkpoint restore
passed on both tasks. Unit suite20/20 passed. Finite losses do not show competence.
Original reports/checkpoints are left untouched.

Independent Phase0 evaluation commands exited without error, but episode reports
have not been supplied to this local workspace. Preflight requires their PASS
reports, expected seeds, zero updates, consistent counts and terminal boundaries.
Previous nvidia-smi output was sampled AFTER evaluation and only showed idle1MiB;
it is not active memory evidence.

New preflight is prepared, not executed: batch256, model5M,<=100 transitions and
3 physical optimizer calls/task (one base, one reference next update, one restored
next update). This is a compatibility check, not a training pilot. Artifacts will
be results/phase1-{push,pick}-preflight-001/{report.json,resolved_config.json,
learner-envelope.pt}. Do not use these weights as a competent nominal checkpoint.

When full training is approved, record per-checkpoint validation table with
train transitions, learner updates, validation transitions, seed-wise success,
return, goal distances, length, losses, gradient norms, planner time, learner time,
memory, wall time, and selected checkpoint SHA256. Report failures without tuning
or automatic restart. Full300000-step completion alone is not a competence pass.
