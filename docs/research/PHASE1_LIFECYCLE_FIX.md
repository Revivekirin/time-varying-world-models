# Phase 1 episode-boundary recreation

## Authorized scope

User explicitly approved the minimal recreation fix and full 300,000-transition
nominal runs for PushCube-v1 and PickCube-v1 after the required gates pass.
Root-cause diagnostic campaigns are stopped. No new pilot, hyperparameter tuning,
upstream changes, dependency changes or tolerance changes are included.

## Implementation

- `src/tvwm/envs/episode_lifecycle.py`: Phase 1-only `EpisodeEnvironment` owns one
  adapter. Before each new episode it synchronizes CUDA, closes the old adapter,
  releases its reference and collects cycles, then constructs a new adapter. It
  rejects resets after stepping an unfinished episode. It compares each new
  episode's contract against the initial contract.
- Construction/cleanup run inside a saved-current-RNG context for Python, NumPy,
  Torch CPU and all CUDA generators, deterministically seeded with the upcoming
  episode seed. The current streams are restored before the ordinary seeded
  reset, never rewound to a previous checkpoint's streams. Actual reset behavior
  remains the adapter's normal behavior. Seed mapping is unchanged.
- `scripts/phase1_train.py` and `scripts/phase1_collector_check.py`: both instantiate
  the same wrapper. Frozen Phase 1 evaluation also uses the same episode lifecycle.
  Agent, optimizers, replay, planner state and counters are not recreated.
- `src/tvwm/agents/nominal.py`: records per-episode recreation and reset seconds.
- `tests/test_episode_lifecycle.py`: verifies ordering, release of old instances,
  mid-episode rejection, current RNG continuity, contract rejection and exception
  handling; server test covers Python/NumPy/CPU/CUDA RNG restoration.
- `scripts/phase1_integrated_launch.sh`: all unit tests (no skipped tests allowed),
  original strict collector checks for both tasks, then two sequential full runs.
  Failure exits immediately, with no retry or diagnostic campaign.
- `configs/phase1/nominal.json`: approval status updated; numerical settings unchanged.

Phase 0 adapter and original artifacts are unchanged. Existing batch-256 preflight
and fresh-process learner-resume artifacts remain valid for model/replay/package
compatibility. Collector evidence must be regenerated because code/lifecycle changed.
The full runner requires the new lifecycle tag and exact code hash. Old full-run
checkpoints, if any, cannot silently resume across the provenance change.

## Validation and launch status

VERIFIED locally: 44 discovered tests, 28 passed and 16 dependency-backed tests
skipped; Python compilation, shell syntax and diff whitespace checks pass.
UNRESOLVED: server unit suite, Push strict collector gate and Pick strict collector
gate. Their original rtol=0/atol=1e-6 comparisons and counter checks are retained.
NOT LAUNCHED: no remote commands were executed by the assistant. The server is
user-operated; the local workstation lacks the server CUDA/ManiSkill runtime.

## Overhead and resources

Actual recreation overhead is UNRESOLVED until the integrated gates run. Each gate
reports warm/reference/child recreation seconds and an INFERRED extrapolation to
6,000 horizon-50 training episodes. Early termination could increase episode count.
Each training episode and validation episode records its measured recreation time.
The former 9–12 h/task estimate excludes this new overhead and is not a new measured
estimate. Existing 24 h/task cumulative limit, 20 GiB used GPU ceiling, 4 GiB free GPU
and 30 GiB free disk floors remain unchanged. No automatic extension or relaunch.

## Execute on server after syncing new files

```bash
conda activate tvwm-phase0
cd /nfs4/jhkim/repos/time-varying-world-models
bash scripts/phase1_integrated_launch.sh
```

This command includes the already-approved full training, conditional on BOTH
gates passing. Run in a persistent terminal. Existing output paths cause an
immediate stop, not deletion or reuse.

| Artifact | New remote path under repository |
|---|---|
| Integrated logs | results/phase1-recreate-workflow-001/ |
| Push collector report | results/phase1-push-collectorcheck-recreate-001/report.json |
| Pick collector report | results/phase1-pick-collectorcheck-recreate-001/report.json |
| Push full run | results/phase1-push-seed1-recreate-001/ |
| Pick full run | results/phase1-pick-seed1-recreate-001/ |

Per-run outputs retain latest.pt, final.pt, interval checkpoints, selected
best-model.pt (and resumable best.pt where available), SHA sidecars, provenance,
resolved configuration, metrics-*.jsonl, validation-*.json, gpu-*.csv and progress.json.
Counters, cadence, optimizer state, replay and cumulative deadline recovery remain
in place. Full commands are in phase1_integrated_launch.sh and remain model_size=5,
batch=256, env=1, seed=1, 300,000 transitions / 299,999 updates per task.

## Remaining failure policy

If a collector gate fails after recreation, stop. Use that integrated report to
identify the first remaining blocker. No automatic relaxation or repeated A/B
campaign. If strict continuation remains unsupported, the smallest defensible
protocol alternative is a separately approved non-resumable fresh run or an
explicit integrity-based continuation protocol; neither is implemented or authorized
by this fix. Training completion alone does not demonstrate competent control.
