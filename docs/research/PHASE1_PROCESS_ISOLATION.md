# Phase 1 spawned simulator isolation

Status: local implementation VERIFIED by dependency-free tests; GPU gates and
50,018-transition checkpoint compatibility UNRESOLVED until the server workflow runs.
Training has not been launched from this workstation.
Local suite: 54 tests, 36 PASS and 18 dependency/CUDA skips. Syntax compilation,
shell syntax and whitespace checks PASS. Server execution must have zero skips.

## Implementation

`src/tvwm/envs/process_env.py` owns a fresh multiprocessing **spawn** worker for
each episode. The child initializes the pinned ManiSkill adapter. The parent
retains the agent, both optimizers, replay, planner state, training RNG and counters.
Observations, actions, rewards, flags, goal distance and RGB frames cross IPC as
CPU arrays/scalars; observation tensors are reconstructed on the parent's CUDA
device. The existing action mapping, seed mapping and update cadence are unchanged.
No upstream or dependency-lock changes are made.

Reset snapshots/restores the CURRENT parent RNG stream. Child construction and
reset use the episode seed. Parent shutdown checks the child's acknowledgement
and zero exit status before replay insertion. Incomplete final-budget episodes
are discarded. Child exceptions, native exits, malformed IPC and 120-second
response deadlines raise errors. Receive deadlines include partial messages.
Failed workers are terminated, joined and, if necessary, killed. Pipes close on
all paths. Linux parent-death SIGKILL also prevents an orphan simulator after a
native learner crash. SIGKILL cannot produce a graceful checkpoint.

`nominal.py`, `phase1_train.py`, `phase1_collector_check.py`, and monitoring use the
same wrapper, including validation videos. The prior in-process wrapper is kept
for historical tests/diagnostics; it is no longer used by Phase 1 training.
Shutdown failure cannot publish an episode to replay. Subsequent attempts must
roll back to the last durable checkpoint, retaining previous metrics.

## Gates and provenance

Run `scripts/phase1_isolation_workflow.sh` on the server once after syncing code.
It runs the unit suite (skips fail the server gate), original Push/Pick collector
checks with **rtol=0, atol=1e-6**, and an exact checkpoint restoration audit. Existing
preflight and learner-resume evidence are reused. No A1/A2/B1/B2 campaign or
training pilot is run. Gate reports include child startup/shutdown memory and
startup overhead; episodes/W&B include shutdown timing too.

The migration script accepts ONLY checkpoint SHA256
`3a29eb658dcf3f97146a3eb856db2330c29353aeea962b543a5fe08998fcc487`,
with transitions=50018, updates=50017, episode_index=1495, boundary=true, and
original code SHA256
`54ea9a962742bf23f7b6fe94aedd5c07d116db2f369bb63b6cf7cc5b3e64418b`.
The latter is computed from the tracked implementation immediately before this
change, not inferred from the checkpoint hash. If the server checkpoint names a
different code version, the audit stops: inspect that version's source before
any migration; do not replace the allowlisted hash just to pass.

It validates pins, package versions, lock hash, configuration, plan, selection
hashes and counters. It restores the actual checkpoint into a fresh learner and
replay, saves it, and compares ALL saved model/optimizer/planner/scale/replay/RNG
and counter state exactly (NaN replay padding compares equal). This uses zero
optimizer updates and zero simulation transitions. Only lifecycle and code-hash
provenance change in a new checkpoint copy. Original artifacts remain unchanged.
`restoration-audit.pt` retains the round-trip evidence. `isolation-audit.json`
records old/new identities. Both collector gates bind the new code hash.

## Execution and outputs

```bash
conda activate tvwm-phase0
cd /nfs4/jhkim/repos/time-varying-world-models
export WANDB_MODE=online
export WANDB_PROJECT=tvwm-phase1
export TVWM_RUN_TAG=isolation-001
bash scripts/phase1_isolation_workflow.sh
```

New paths:
- `results/phase1-workflow-isolation-001/`: unit/gate/audit/training logs.
- `results/phase1-{push,pick}-collectorcheck-isolation-001/`: strict collector evidence.
- `results/phase1-push-resume-isolation-001/`: audited checkpoint copy, preserved
  wall ledger and selection artifacts, subsequent metrics, videos, checkpoints.

Only after all checks pass does this workflow resume Push from the copied
checkpoint with `--resume --acknowledge-rollback --approve-budget 300000`.
No fresh Push weights or Pick training are launched by this recovery workflow.
No retries or overwrites. Existing output directories are preserved.

The original 24-hour cumulative wall ledger is copied unchanged. Its existing
policy charges unclosed-attempt downtime. If exhausted, the workflow reports
NO_GO and does not reset the ledger or grant more time. Runtime projection uses
observed checkpoint wall time/episode length and new startup overhead; it is
INFERRED, conservative, and not a promise of finishing within the remaining
budget. The production runner enforces the original time/storage/GPU guards.

## Remaining limits

Local tests cannot measure RTX3090 startup/memory, validate GPU trajectories, or
read server-only checkpoints. Bounded collector checks cannot prove arbitrary
long-run simulator stability. Process startup may materially reduce throughput.
Cross-lifecycle migration is explicitly recorded; it does not promise equality
to an uninterrupted run of the old unsafe implementation. Nominal competence,
final training success and Phase 2 authorization remain unestablished.
