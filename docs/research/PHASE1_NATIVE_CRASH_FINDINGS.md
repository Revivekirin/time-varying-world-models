# Phase 1 native lifecycle crash findings

## VERIFIED — supplied server evidence

- Training attempt 1 ended after progress episode_index=935.
- Resume restored episode_index=764 and ended after progress episode_index=1700
  (936 additional completed episodes). Last checkpoint is transitions=50018,
  updates=50017; SHA256 3a29eb658dcf3f97146a3eb856db2330c29353aeea962b543a5fe08998fcc487.
- The lifecycle-only diagnostic made no environment step, planner call or
  optimizer update and did not initialize W&B. It independently segfaulted.
- Cycles 0 through 940 completed: 941 successful creation/reset cycles. Cycle 941
  (942nd attempt) entered adapter.construct.begin and its internal gc.collect
  returned. adapter.construct.end was not recorded. The process emitted a native
  segmentation fault. report.json was not reached, as expected for native death.
- At cycle 939 the sampled device usage was 1450 MiB, CUDA free 23917494272 bytes,
  RSS 2029788 kB, threads 23 and fd_count 318. These are samples, not peak values
  or proof against all memory corruption/resource failures.

## INFERRED — sufficiently localized engineering blocker

Repeated in-process ManiSkill/SAPIEN environment lifecycle is sufficient to
reproduce a native crash near the observed training lifetime. Learner execution,
physics stepping and W&B logging are not necessary to reproduce this crash.
The last entered operation is construction; without a native stack this does not
prove the corrupting operation occurred there, or identify a specific SAPIEN,
PhysX, CUDA, rendering or GC defect. GPU-capacity OOM is not supported by the
available samples; native leaks, handle exhaustion or memory corruption remain
possible. Do not label a specific internal library bug as established.

## Readiness and proposed corrective boundary

Current production in-process recreation: NO-GO for another long run. Preserve
all checkpoints, logs, lock files and the strict continuation tolerance. No more
repetitions of the same 1000-cycle crash campaign are needed.

The preferred correction is simulator process isolation: keep the learner,
optimizers, replay, counters and training RNG in the parent; create a fresh spawned
simulator worker per episode, exchange only observation/action/reward/flags and
required metrics, and terminate/join each worker before the next episode. Do not
fork an initialized CUDA process. This avoids cumulative native scene recreation
inside one process without weakening episode freshness or changing learning
objectives. It is a proposed fix, NOT implemented or verified in this report.

Necessary acceptance: same contract and seed mapping, RNG/cadence tests, unchanged
strict Push/Pick collector gates, measured process overhead/resource guards, and
an explicit audited provenance migration if continuing the existing checkpoint.
Do not remove code-hash checks, rewrite old provenance or silently start from new
weights. Process isolation may cost materially more startup time; the existing
24-hour budget cannot silently expand. Native root cause remains UNRESOLVED even
if process isolation removes the observed operational failure.
