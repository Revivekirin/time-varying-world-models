# Phase 1 resource usage

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

## VERIFIED measurements

RTX 3090, 24 GiB. Batch-256 preflight device sampled peak: Push 1930 MiB,
Pick 1886 MiB. Torch allocated/reserved peaks: Push 455352320/484442112 bytes;
Pick 379809280/438304768 bytes. Approximately 21.8 GiB CUDA free after tests.
Storage free at preflight about 722–728 GB; quotas remain UNRESOLVED.
Fresh-process learner-only resume allocated/reserved: Push 268188672/308281344;
Pick 276627456/316669952 bytes. These are not full training peaks.

## INFERRED full-run estimate

Phase 0 simulation and warm planner timing plus measured warm batch-256 updates
imply baseline approximately 7.81 h Push / 7.35 h Pick. With validation, checkpoint
I/O and contention, plan approximately 9–12 h per task; runtime remains UNRESOLVED.
Cold first-update latency is not used as steady-state throughput. Run tasks
sequentially. Full-run maximum is 24 h per task, with an internal 900 s shutdown
reserve and external timeout. No automatic restart or budget extension.

Launch and periodic guards require device used <=20 GiB, CUDA/device free >=4 GiB,
and filesystem free >=30 GiB. Single physical GPU mapping is required; ambiguous
multi-GPU hosts fail closed. nvidia-smi runs concurrently every second to CSV;
structured memory samples are written after episodes and every 100 updates.
Torch peak allocated/reserved are recorded. Samples cannot exclude subsecond
spikes. Validation worker also records its own Torch peaks. No batch/model fallback.

Checkpoint interval 10,000 transitions (next completed boundary), plus first
completed episode latest, best, final and stop snapshots. Each contains full
replay. Check free disk before publication. Peak storage and full-run speed must
be reported from actual results; estimates are not reported as measurements.

Interrupted attempts use a persistent wall ledger; unclosed attempts conservatively
charge elapsed downtime on resume. Graceful resume retains cumulative wall usage.
The external timeout allows 120 s forced-kill grace; this is emergency shutdown
overhead, not additional authorized training time.
