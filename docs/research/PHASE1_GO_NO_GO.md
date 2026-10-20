# Phase 1 go/no-go

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

As of the user-supplied fresh-process learner reports: **NO-GO until the new
collector/evaluation check passes and the user explicitly approves the full budget.**
No full training has been executed by this agent.

| Gate | Status |
|---|---|
| Phase 0 contracts, replay, model/planner/learner and frozen evaluation | VERIFIED by supplied reports and preflight |
| Batch 256 forward/backward/optimizer, full replay storage | VERIFIED on both tasks |
| Active GPU headroom and available disk at preflight time | VERIFIED; rechecked at launch and during training |
| Official warmup/update cadence and episodic handling | VERIFIED against pinned source |
| Same-process and fresh-process learner restore | VERIFIED on both tasks |
| Shared collector episode-boundary continuation in a fresh process | UNRESOLVED pending phase1_collector_check.py |
| Full runner frozen evaluator compatibility | UNRESOLVED; included in that check with two seeds |
| Full runner CUDA end-to-end execution | UNRESOLVED; implementation supplied, local GPU unavailable |
| Explicit 300,000-transition/task budget approval | Required via user-executed --approve-budget 300000 |
| Nominal competence and stability | UNRESOLVED; no full-run results yet |

The runner checks task identity, evidence statuses, checkpoint hashes, exact
source/package/lock provenance, code/config fingerprint, actual environment
contract, free memory and storage before training. Any affected failed gate stops
that task. Run the two bounded checks first; do not combine them with an automatic
full-training launch. The explicit launch command is a separate user decision.

No 10,000-transition pilot, dynamics shifts, history inputs, online adaptation,
changed objectives, changed planning hyperparameters, or automatic Phase 2.

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
