# Phase 1 nominal results

**UNRESOLVED: no full nominal training result has been supplied or executed here.**
There are no competent nominal checkpoints to claim yet.

**VERIFIED:** Phase 0 contract and learner integration passed for both tasks.
Independent two-seed Phase 0 frozen evaluation had 0/2 successes for each task;
Push returns 6.228174 / 5.174648; Pick 1.463695 / 1.547711. These short smoke-trained
models establish operation, not nominal competence. Both batch-256 preflight and
fresh-process learner continuation passed. Latest check adds zero transitions
and one learner update per task, not full nominal training.

**UNRESOLVED:** next bounded collector/evaluator test; subsequent full training,
validation success distributions, goal progress, reward/Q/model losses, planner
latency, learner throughput, peak memory, and final selected checkpoint hashes.
The new runner emits attempt-specific metrics JSONL, validation episode JSON,
concurrent GPU CSV, resolved configuration, provenance and checkpoint sidecars.
No test output is invented. Local checks passed 22 tests; eight dependency-backed
tests were skipped because this macOS environment has no server tensor stack.

Report across-evaluation-seed success and return distributions, terminal success,
goal progress and failures. Report training-seed count as one per task initially.
Heldout evaluation of the already-selected best model must not feed selection.
A completed transition budget does not establish nominal competence. Diagnose
integration, reward, replay and optimization if control is weak; no Phase 2 starts
automatically. Original Phase 0 files and artifacts are unchanged.
