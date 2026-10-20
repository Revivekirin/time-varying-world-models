# Phase 1 checkpoint audit

## Current lifecycle integration status

The user has explicitly approved the full nominal budget conditional on gates.
Episode-boundary environment recreation is now implemented identically in Phase 1
collection, collector verification and evaluation. Root-cause campaigns are stopped.
Local checks: 28 passed, 16 dependency skips. Server collector gates and recreation
overhead are UNRESOLVED; training is NOT LAUNCHED by the assistant. Existing preflight
and learner-resume evidence is reused; strict collector evidence must be regenerated.
Both gates must pass before the integrated script launches sequential full runs.
See [PHASE1_LIFECYCLE_FIX.md](PHASE1_LIFECYCLE_FIX.md) for the current implementation,
commands, output paths and limitations. Older statements below describe historical
states and do not revoke the explicit budget approval or grant an unpassed gate.

**VERIFIED (user-provided server reports):** both tasks passed
PASS_LEARNER_RESUME_ONLY in a fresh process. Each restored transitions=100,
updates=1, episode_index=2 and executed exactly one further batch-256 update,
matching the reference next loss. This is not full training or simulator restore.

| Task | Preflight envelope SHA256 | Fresh update seconds |
|---|---|---:|
| PushCube-v1 | 5c06a8f0e7d6e973b132b38d055068aeefb9ac0ed1b1f585afe6f135430c3b78 | 1.56734 |
| PickCube-v1 | 0e83eaaff29574b6d49df432a8895c81c3f34f35239966889fb7b1613f938180 | 1.57870 |

**UNRESOLVED:** the new collector check reuses the actual training collector,
collects a warm episode, saves an envelope at a boundary, then compares the next
reference episode to a fresh process loading that envelope. Observations,
actions, rewards and flags must agree at absolute tolerance 1e-6; counters must
match exactly. It also exercises the full runner's frozen evaluator on two seeds.
Maximum 250 simulation transitions/task, zero optimizer updates. This is a
compatibility check, not a training pilot. Any mismatch fails the launch gate.
No general bitwise GPU determinism claim follows from a passing check.

The gate report binds code/config SHA256 and the original preflight envelope hash.
Full-run checkpoint metadata records SHA256, counters and replay device; envelope
provenance binds resolved configuration, sources, dependency lock and packages.
Only trusted local checkpoint files may be loaded (torch weights_only=False).

**Implemented, runtime UNRESOLVED:** interval/full-run envelope saving and resume,
separate latest/final, validation best-model and best resumable envelope where
available. If best is the initial model, best-model.pt is its exact artifact and
there is no resumable best.pt. Metadata/hash mismatch fails closed, including a
crash between checkpoint publication and sidecar publication. Use a separately
verified interval snapshot only through an explicit recovery procedure; do not
silently substitute checkpoints. Phase 0 artifacts remain untouched.

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
