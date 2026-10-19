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

## Collector check 001 server evidence

VERIFIED: all 30 server unit tests passed without skips. Push collector check
001 failed while appending the first collected episode to restored CUDA replay:
incoming episode tensors were on CPU. No optimizer updates ran. Pick was not
executed because the shell used set -e. Full training remains NO-GO.

The shared Phase 1 collector now explicitly transfers new episodes to an already
initialized replay storage device before insertion. This changes device placement
only, not replay alignment, rewards, objectives, model or batch size. Regression
tests cover serialized/restored TorchRL CPU and CUDA storage followed by append.
The runtime fix is UNRESOLVED until the server tests and collector checks pass.
Retry Push with collectorcheck-002; Pick can use collectorcheck-001. Old failed
artifacts and all Phase 0 artifacts remain unchanged.

## Collector check 002 server evidence

VERIFIED: all 32 server tests passed, including restored CPU/CUDA replay append.
Push fresh-process rollout comparison failed: 1322/1785 observation elements
differed at atol=1e-6, maximum absolute difference 4.558298110961914.
Zero optimizer updates ran; Pick was not executed. The insertion fix passed,
but collector continuation remains NO-GO. Cause is UNRESOLVED: the original
report did not distinguish initial-state, first-action or later trajectory drift.
The diagnostic revision saves reference.pt and child.pt and reports each trace's
first mismatch, first-row difference, shapes and counters before failing.
Tolerance and training configuration are unchanged. Retry Push collectorcheck-003
only before deciding on further corrective work.
