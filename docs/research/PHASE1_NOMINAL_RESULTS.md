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

## First preflight attempt: contract serialization mismatch

User-run Push preflight001 failed before any transitions/updates on action_mapping
comparison. Runtime tuple ranges versus JSON list ranges were compared directly.
Comparison now normalizes JSON representation, retaining checks on actual indices,
scalar changes and missing fields. Four local regression cases pass. No algorithm,
controller, batch or resource budget changed. Pick preflight did not execute.

This report independently verifies Push Phase0 frozen evaluation: seeds10001/10002,
50 transitions each, returns6.228174/5.174648,0/2 successes,0 updates,5.781759s.
This is operational evaluation evidence, not nominal competence. Dependency pip
check passed. Free filesystem bytes718346911744. Sampled GPU peak1450MiB occurred
only during initialization, not batch256 learning; compatibility remains unresolved.
New bounded retry uses push-preflight002 and initial pick-preflight001 outputs.

## Both batch256 compatibility gates passed

VERIFIED user-provided server2 reports: Push preflight002 and Pick preflight001
PASS_COMPATIBILITY_ONLY. Each100 transitions/3 physical optimizer calls;
logical restored count2. Unit suite27/27 passes in3.520s. Checkpoint frozen
forward, terminated bootstrap mask, finite batch256 forward/backward/optimizer,
replay300000-row allocation and same-process learner restore all passed.
Evidence archived in reports/phase1-preflight-evidence.json without changing
Phase0 files. Independent Phase0 evaluation also verified for Pick: two50-step
truncated episodes, returns1.463695/1.547711,0/2 successes. No competence claim.

New bounded fresh-process learner audit: scripts/phase1_resume_check.py,
manifest configs/phase1/resume_check_manifest.json. Per task0 transitions,
0 simulator constructions,1 batch256 optimizer call,180s internal deadline.
Checks source/package/lock/checkpoint hashes, empty replay storage restoration,
counters and next-update metrics against saved uninterrupted reference.
Prepared and syntax-checked only; remote result pending. Collector continuation
and full-run runner remain separate incomplete requirements; launch not approved.
