# Phase 1 checkpoint audit

VERIFIED upstream save: model state only, including target Q parameters. It does
not contain optimizer, replay, running actor-Q scale or collector/RNG counters.
It cannot be called a resumable training checkpoint. Phase0 model save/reload and
seeded frozen inference passed; Phase0 snapshots remain unchanged.

Prepared learner envelope src/tvwm/agents/resume.py stores model, both Adam
optimizers, running scale, planner mean, upstream replay state_dict and episode
counter, supplied logical counters, source/config provenance and Python/NumPy/
CPU/CUDA RNG. Writes temp file, fsync, atomic replace. Loads trusted local files
only and requires exact provenance. Buffer storage must first be initialized.

UNRESOLVED runtime gate: phase1_preflight saves after one update, performs a
reference update, restores envelope, repeats next update, compares metrics and
model state. This verifies same-process learner restoration only. It must pass
before use, and does not prove collector/fresh-process resume correctness.

Required full-run collector envelope (not yet implemented/verified): completed-
episode boundary, exact env-transition count, learner count, phase within initial
1000-update burst, next episode seed/index, training/eval counters, next checkpoint/
eval due step, best validation score, cumulative wall time and append-only event
sequence. Save only after the completed episode is inserted and all scheduled
updates for that transition finish. Resume before next reset using explicit
per-episode seed; do not pretend native get_state_dict restores elapsed time/RNG.
Record any physical work lost after last durable checkpoint separately from
logical resumed counters. No silent counter/schedule reset or redoing the initial
burst. Validate clean uninterrupted versus fresh-process checkpoint continuation
at the1000/1001 boundary and a normal episode boundary before launch.

Predefined retention: latest resumable checkpoint, best validation checkpoint and
final snapshot separately. Keep small manifests/hashes/validation records for all
intervals. Model-only evaluation snapshots may accompany resumable envelopes.
Disk/quota checks and atomic publication must cover replay snapshots; retain the
previous latest until replacement succeeds. Final partial episode accounting must
be explicit, never create a false terminal label to hit a budget.

Each manifest must include task, training seed, source Git hashes, local project
revision/diff hash, dependency lock hash, resolved config hash, checkpoint hash,
training/evaluation counters, checkpoint kind, validation metrics and wall time.
Preflight envelopes are diagnostic only and are not sufficient to launch/resume
full training until collector and fresh-process gates pass.
