# Phase 1 throughput optimization — candidate, not approved deployment

**Verdict: UNRESOLVED / NO-GO for optimized training.** Current training was not
contacted, stopped, relaunched or modified. No server commands were executed.
The local machine cannot access `/nfs4/jhkim/repos/DRL`, the running process or its
latest checkpoint. Do not overwrite the running checkout with these edits: its
future spawned workers/evaluation processes import files from disk.

## Source audit

VERIFIED from the current TVWM checkout and its pinned TD-MPC2 source:

| Path and line | Behavior and consequence |
|---|---|
| `src/tvwm/envs/process_env.py:20` | Each episode creates a spawned Python process and Pipe. |
| `src/tvwm/envs/process_env.py:33` | Every IPC receive creates a new thread; the timeout covers partial messages. Removing that protection merely to reduce overhead is unjustified. |
| `src/tvwm/envs/process_env.py:87` | Child imports Torch/adapter, seeds, constructs a new GPU scene and resets. Two nvidia-smi subprocesses per episode enforce/record memory at startup and shutdown. |
| `src/tvwm/envs/process_env.py:120` | Observation CUDA → CPU NumPy copy; parent sends CPU action list; child converts action back to CUDA. |
| `src/tvwm/envs/process_env.py:185` | Synchronous request/response per transition; parent observation converted back to CUDA. Small 35/42-float observations mean bandwidth alone is unlikely to dominate (INFERRED). |
| `external/tdmpc2/tdmpc2/tdmpc2.py:98` | Official act transfers observation to learner CUDA, calls stochastic planner and returns CPU action. Its CPU action transfer is already performed upstream; changing planner output/state is not part of this candidate. |
| `external/tdmpc2/tdmpc2/tdmpc2.py:163` | Sampling/planning uses GPU tensors, RNG, and previous planner mean. Keep t0 and planner state unchanged. |
| `src/tvwm/agents/nominal.py:44` | Planner and each update are synchronized/timed; scalar metrics also synchronize. Current timing is inclusive, not pure kernel time. |
| `src/tvwm/agents/cadence.py:2` | 1,001 random transitions; 1,000-update burst at zero-based step 1,000; then one update/transition. Exact full cap yields 299,999 updates. |
| `src/tvwm/agents/nominal.py:69` | Terminal observation retained; only a complete episode whose worker closes successfully enters replay. Partial final-budget episode is discarded. |
| `external/tdmpc2/tdmpc2/common/buffer.py:17` | Episode SliceSampler uses horizon+1 rows. Storage CUDA/CPU is selected separately from learner CUDA. Replay extends completed episodes, not vectorized individual samples. |
| `external/tdmpc2/tdmpc2/tdmpc2.py:334` | Update samples replay and executes the official objectives. Unmodified. |
| `scripts/phase1_train.py:210` | Checkpoint publication serializes full state and hashes it. Validation is synchronous in a separate process, with independently seeded episodes and videos. Both pause training transitions. |

INFERRED: repeated CPU imports and scene initialization are important when success
terminates episodes after few steps. Existing logs do not quantify their separate
shares. This is not proof that IPC or physics dominates. No synchronization,
reward, UTD, reset, model, batch or vectorization change is justified from the
available timing evidence.

## DRL comparison

The supplied task states DRL uses ManiSkill 3.0.1, 32 GPU environments and SAC/MBPO.
These are user-provided facts, **not independently VERIFIED from DRL source here**.
`throughput_inventory.py` collects commit/file hashes and numbered source context
for review without importing DRL. It does not label keyword hits as an audit.

| Topic | TVWM verified | DRL source verdict / transfer decision |
|---|---|---|
| Construction/reuse | One fresh GPU simulator per episode | UNRESOLVED; persistent simulator reuse cannot transfer until warm–fresh contract is resolved. |
| Vectorization | num_envs=1 | 32 reported; switching violates approved configuration and changes data/update/planner handling. |
| Transfers/device | CPU IPC between separate CUDA-owning processes | UNRESOLVED; direct in-process GPU tensor sharing cannot be copied blindly. |
| Replay/UTD | Completed episode slices, exact pinned update cadence | SAC/MBPO reported; exact insertion/UTD UNRESOLVED, not transferable by assumption. |
| Auto-reset/terminal observation | Explicit reset; final observation saved before close | UNRESOLVED; must inspect final-observation and truncation handling. |
| Horizon/controller | Horizon50, Panda delta joints, 100Hz/20Hz | UNRESOLVED; no cross-framework FPS comparison until matched. |
| TD-MPC2 presence | Exact pinned source inspected locally | UNRESOLVED in DRL; no claim that SAC is TD-MPC2. |

## Small, opt-in candidate

`src/tvwm/envs/prefetch_env.py` overlaps only CPU module imports for the next
worker. It asserts Torch CUDA is not initialized during preparation. Actual
ManiSkill import/construction, CUDA initialization, reset and episode seed
application remain behind an activation message, sent after the prior GPU worker
has been closed and joined. There is at most one active GPU simulator plus one
CPU preparation process. Parent RNG is snapshot/restored around preparation;
learner/planner/optimizer/replay are never transferred to that child.

CPU preparation can increase RSS and CPU contention. It does not remove CUDA or
scene startup cost. Default pipeline remains `baseline`; `cpu-prefetch` is an
explicit CLI choice propagated through original collector child/evaluation
processes and training validation. Lifecycle provenance differs. No upstream or
lock changes. No production IPC/synchronization changes are bundled with it. Full training with
`cpu-prefetch` is explicitly blocked in the runner until performance acceptance
and latest-checkpoint migration are verified; evaluation/gate usage remains available.

Modified: `phase1_train.py`, `phase1_collector_check.py` (explicit pipeline and
provenance binding). Added: `prefetch_env.py`, `test_prefetch_env.py`,
`throughput_inventory.py`, `throughput_compare.py`, this report.

## Evidence and performance acceptance

Local VERIFIED: 59 tests, 40 PASS, 19 dependency/CUDA skips; syntax compilation
and diff checks PASS. Four dependency-free candidate tests cover prepare/activate protocol,
explicit selection, preserving prepared worker until boundary and cleanup on
active-worker shutdown failure. An additional server-only test checks current RNG preservation and seed delivery across consecutive prepared resets. Existing no-incomplete-replay tests remain.
GPU cleanup, actual candidate RNG equality and Push/Pick gates: UNRESOLVED.

| Measurement | Existing evidence | Candidate |
|---|---|---|
| Whole-attempt transitions/s including eval/save | INFERRED ~2.98 from user log (50,018→60,242 in57m14s) | UNRESOLVED |
| Transitions/s excluding eval/save | Not derivable exactly from old component timers | Bounded comparison records explicit total time |
| Learner updates/s | Existing update metric duration sums available remotely | No new learner update in frozen comparison; UNRESOLVED |
| CPU import / activation | Previously combined startup timer | Separate candidate preparation and activation timers |
| Episode wall time | Old episode component sums are not complete wall time | Explicit enclosing wall timer |
| Planner | Existing synchronized planner_seconds | Same collector/timing |
| Step IPC | Not separately recorded | Transport round-trip wall includes physics, copies and IPC; not pure IPC |
| Pure physics | UNRESOLVED | UNRESOLVED; no intrusive GPU instrumentation added |
| Shutdown, memory, RSS, worker lifetime | Shutdown/GPU metrics in remote JSON | Benchmark records startup/shutdown GPU metrics, parent/child peak RSS, process lifetime and pipe/join outcome |
| Checkpoint/evaluation | Blocking paths verified; full timings not yet obtained | Excluded from frozen comparison; protocol unchanged |

`throughput_compare.py` runs baseline/candidate/candidate/baseline in independent
processes, two complete episodes each: at most400 transitions/task, ZERO optimizer
updates. Identical checkpoint, seeds, RNG, frozen planner, replay starting state,
GPU, video policy (none), and host timing wrapper. Outputs are diagnostic only;
no learner checkpoint is published. Compare final counters/RNG exactly and trace
values at rtol0, atol1e-6. Each child has a300s timeout; no retries. GPU processes
present => refuse to start. This is not a training pilot.

Frozen-collection gain <10% => REJECT_NEGLIGIBLE_GAIN. Gain >=10% only means
PROMISING_FROZEN_COLLECTION_ONLY. Two paired trials do not establish robust
full-training speedup, especially CPU contention with learner updates. Do not
promote on instantaneous tqdm ETA or extrapolate frozen-policy FPS as training
FPS. Production acceptance additionally needs both original gates, actual DRL
review, and checkpoint/budget audit. No speedup is currently claimed.

## Checkpoint continuity

Latest >60k checkpoint identity is UNRESOLVED locally. Inventory checks the live
latest hash/sidecar consistency but does not load, rewrite or migrate it. A stable
immutable checkpoint must be rechecked after current training ends. The earlier
50,018-only migration tool is NOT applicable to a later checkpoint and must not
be rerun for this optimization.

The new code hash and optional lifecycle differ: existing checkpoint cannot pass
the unchanged exact provenance loader as-is. Verdict: **no optimized resume yet**.
If candidate is accepted, an explicit new-copy migration must bind original hash,
old/new code manifests, identical config/source/package locks, both optimizer and
replay contents, RNG, planner state, counters, selection artifacts and wall ledger.
Never edit existing metadata to satisfy that comparison. Preserve24h cumulative
budget including existing downtime policy. Budget exhaustion is not permission
to grant a new allocation. Pick remains gated separately.

## Exact next commands

First upload the candidate archive to `/tmp/tvwm-throughput-candidate-001.tar.gz`.
Prepare a separate checkout; **do not sync over the live repository**:

```bash
conda activate tvwm-phase0
live=/nfs4/jhkim/repos/time-varying-world-models
candidate=/nfs4/jhkim/repos/time-varying-world-models-throughput-001
set -e
mkdir "$candidate" && cp -a "$live/src" "$live/scripts" "$live/configs" "$live/environment" "$live/tests" "$candidate/" && ln -s "$live/external" "$candidate/external"
# Only continue if the exclusive mkdir/copy above succeeded.
cd "$candidate"
tar -xzf /tmp/tvwm-throughput-candidate-001.tar.gz
python scripts/throughput_inventory.py \
  --run "$live/results/phase1-push-resume-isolation-002" \
  --drl /nfs4/jhkim/repos/DRL \
  --out results/throughput-inventory-001
```

Inventory is the recommended next action while training continues. Return its
`inventory.json` for actual DRL and current-budget interpretation. Run-directory
name follows the last isolation-002 command; adjust only if your active run has
a different path.

Only after current GPU work ends, in the separate checkout:

```bash
set -e
python scripts/run_server_tests.py
# Use the immutable checkpoint observed in the pasted log, if still available.
# Never substitute a different file without recording its SHA/provenance.
timeout --signal=TERM --kill-after=15s 1250s python scripts/throughput_compare.py \
 --checkpoint "$live/results/phase1-push-resume-isolation-002/checkpoint-000060014.pt" \
 --out results/throughput-push-001
timeout --signal=TERM --kill-after=15s 1250s python scripts/throughput_compare.py \
 --checkpoint "$live/results/phase1-pick-preflight-001/learner-envelope.pt" \
 --out results/throughput-pick-001
```

Push uses the trained policy; Pick uses smoke weights and is NOT evidence of
trained Pick throughput. Stop if gain is negligible; do not repeat the comparison.
If promising, run the original gates once:

```bash
set -e
timeout --signal=TERM --kill-after=15s 600s python scripts/phase1_collector_check.py \
 --preflight "$live/results/phase1-push-preflight-002" \
 --resume-check "$live/results/phase1-push-resumecheck-001" \
 --simulator-pipeline cpu-prefetch --out results/push-prefetch-gate-001
timeout --signal=TERM --kill-after=15s 600s python scripts/phase1_collector_check.py \
 --preflight "$live/results/phase1-pick-preflight-001" \
 --resume-check "$live/results/phase1-pick-resumecheck-001" \
 --simulator-pipeline cpu-prefetch --out results/pick-prefetch-gate-001
```

No new full-training command is recommended yet: the running baseline continues,
performance acceptance and latest-checkpoint migration have not passed. This is
an explicit readiness limitation, not authorization to restart from fresh weights.
