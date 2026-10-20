# Phase 1 nominal training configuration

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

## Proposed budget and authorization

One seed (1) each for PushCube-v1 and PickCube-v1; nominal dynamics only.
Official pinned TD-MPC2 model_size=5, batch_size=256, environment count=1,
300,000 actual environment transitions per task. No separate pilot.
`configs/phase1/nominal.json` is the proposal; executing the full command with
`--approve-budget 300000` records explicit approval for the task and its limits.
Mandatory gates must pass first. No remote command is executed by the agent.

## VERIFIED upstream compatibility

Pinned official online trainer uses zero-based steps, random actions through
step 1000 inclusive (1,001 transitions), then policy actions. After transition
1001 it performs 1,000 optimizer updates; subsequent transitions get one update.
The exact 300,000-transition cap therefore gives 299,999 learner updates.
The official inclusive `while step <= steps` would collect one extra transition;
the runner changes only that endpoint and documents it explicitly.

Updates sample only previously completed replay episodes. Current episode enters
replay after its final update. Rows are o0/NaN placeholders followed by
(o[t+1], a[t], r[t], terminated[t]); truncation closes episodes but bootstraps.
Strict upstream sequence sampling and objectives remain unchanged.

Native float32 state observations: Push 35, Pick 42; normalized eight-dimensional
Panda pd_joint_delta_pos actions; normalized_dense rewards; horizon 50;
100 Hz physics, 20 Hz control, five physics substeps per transition. No repeat.
The action contract and controller configuration are checked against preflight.
Episodic termination head is enabled as previously audited. Compile remains false.

## Collection, validation and selection

Training reset seed is 1,000,000 + 100,000 * training_seed + episode_index.
This explicit per-episode seed schedule enables boundary restart and is disjoint
from validation seeds 11001–11020. Initial validation and validation at the first
completed episode boundary at/after every 10,000 transitions; also at exact end.
Validation runs in a separate process with frozen model and independent RNG.
Up to 31,000 validation transitions/task are additional to training transitions.
No validation data enters replay. Success is any success in an episode; terminal
success is also recorded. Selection orders validation success rate, mean return,
then earliest checkpoint on ties. Goal progress is initial minus final object-to-
goal distance (Push XY; Pick XYZ), a diagnostic only; reward is unchanged.

Latest, interval, final and best models are separate. Best-model.pt always holds
the selected model; best.pt additionally holds a resumable envelope if the selected
validation happened after replay initialization. An initial untrained winner has
only best-model.pt, not an invented initialized replay checkpoint.
The fixed heldout seeds 21001–21100 are evaluated only after selection and never
used to select best. One training seed cannot establish across-training-seed stability.

## Restart contract

Checkpoints store model, both optimizers, running scale, planner mean, full replay,
RNG, transition/update/episode counters, next validation/checkpoint thresholds,
selection score, evaluation steps, wall time and provenance. Save at episode
boundaries after all updates and replay insertion. Latest is first saved after
the first episode, then at interval boundaries and graceful stop. No mid-episode
simulator state restoration is claimed. Full training initializes new weights;
preflight/smoke checkpoints are only compatibility evidence.

If the exact end budget cuts an episode, it is not fabricated into a completed
replay episode. Final records boundary=false and cannot resume. No further
training beyond the approved cap is supported automatically.

On SIGTERM/SIGINT finish the current episode and save. Hard failures preserve the
last durable checkpoint. Resume requires --acknowledge-rollback, records the prior
observed counters and restored counters, and preserves attempt-specific metrics.
Physical work since the last durable checkpoint may repeat; it never silently
changes the restored schedule. A wall ledger charges failed attempt time; after
an unclosed process it conservatively includes downtime until restart. Code,
config or dependency drift blocks resume; hyperparameters are never auto-adjusted.
