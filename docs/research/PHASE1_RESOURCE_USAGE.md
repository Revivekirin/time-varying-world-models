# Phase 1 resource usage

VERIFIED Phase0 user measurements, RTX3090 24GiB, Torch2.7.1+cu118:

| Task | Adapter step/s | MPPI mean excluding first call | Batch16 update/s excluding first | Torch peak allocated/reserved |
|---|---:|---:|---:|---:|
| Push |67.568|35.696ms|31.421|292948992 /360710144 bytes|
| Pick |63.220|34.809ms|26.779|293184512 /362807296 bytes|

First update costs1.339s/1.311s, first planner143.7ms/70.8ms. Tiny samples, batch16,
and adapter instrumentation limit extrapolation. Torch memory excludes SAPIEN.
The1MiB idle nvidia-smi sample does not establish running device usage.

INFERRED illustrative batch16-based extrapolation:300000*(step time + warm MPPI
latency + warm update time) gives roughly6.9h Push and7.3h Pick before evaluation,
checkpoint I/O and batch256 cost. These are not batch256 runtime predictions or
guarantees. Replace optimizer term with measured batch256 timings after preflight:
T =300000*step_time +298999*plan_time +299999*update_time + validation + I/O.
Validation upper bound31000 transitions/task plus final heldout5000. Serial runs
avoid simultaneous training/evaluation simulator memory peaks.

Proposed budget:20GiB device-wide with4GiB CUDA free headroom,30GiB free output
storage,24h cumulative wall/task. Final wall budget approval follows measured
batch256 timings. Replay300000 rows has roughly55–64MiB raw payload at observed
state/action dimensions including scalar fields/episode IDs; allocator, model,
optimizer, learner activations, simulator and serialization add overhead.
Do not extrapolate the tiny Phase0 replay memory directly to full training.

Preflight allocates upstream replay at proposed300000-row capacity, uses batch256
and records torch peak allocated/reserved, cuda.mem_get_info before/after, sampled
nvidia-smi device used/free/total while test runs. Samples every~.2s can miss narrow
peaks; report as sampled device peak, not exact peak. No separate-terminal timing
needed. Records local filesystem free capacity (NFS user quota remains UNRESOLVED).

Do not silently shrink batch/model, offload differently from upstream heuristic,
raise memory ceiling, or auto-relaunch. A budget failure blocks that task.
Installed package/build lock and active-process provenance must accompany full
run. Batch256 compatibility test has600s internal/630s external deadline/task,
max100 transitions and3 optimizer calls. No full training has run.
