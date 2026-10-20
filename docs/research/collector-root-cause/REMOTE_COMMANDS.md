# Bounded user-run investigation

Sync the new diagnostic script and tests to the server. Keep original artifacts.
The script writes only to a new output directory and compares hashes of the three
input artifacts before/after. It never launches training, optimizes parameters,
patches production/upstream code or grants launch readiness.

```bash
conda activate tvwm-phase0
cd /nfs4/jhkim/repos/time-varying-world-models
(
set -e
PYTHONPATH=src python -m unittest discover -s tests -v
timeout --signal=TERM --kill-after=15s 1860s \
  python scripts/phase1_root_cause.py \
  --source results/phase1-push-collectorcheck-003 \
  --out results/phase1-push-rootcause-001
)
```

Expected maximum: 200 simulation transitions, 250 planner calls, 0 updates.
Workers run sequentially. Each has a 180-second timeout and its own log.
Comparisons can FAIL while the diagnostic itself completes successfully; this
never indicates that Phase 1 is ready. A failed worker is logged while other
independent jobs continue. A2 is explicitly UNRESOLVED without original warm data.

Share summary.json and root-cause-report.md. If a worker fails, share its .log.
Per-step details are in A1-per-step.json, A1_probe-per-step.json,
probe_effect-per-step.json, B1-per-step.json, B1_all_t0-per-step.json and
B2-per-step.json. Raw snapshots and CPU/all-CUDA RNG tensors remain in worker .pt
files; no need to paste binaries. The generated source-references.json binds
paths and line numbers to the files actually used on the server.

Optional A2 uses --warm-history PATH with an existing original recording manifest:

```json
{
  "original_recording": true,
  "episodes": [
    {"path": "original-warm.pt", "sha256": "ACTUAL_SHA256", "seed": 1100002}
  ]
}
```

The path is relative to the manifest directory. File format is
`{"trace": {"actions": Tensor[T,8], ...}}`; T <= 50. The seed is checked against
boundary episode_index minus one, not accepted blindly. This must be original
recorded data from the preceding warm episode, not newly generated actions or an
unrelated replay episode. Default runs do not invent such a file. A2 adds at most
100 environment transitions. Use a new output directory for any added run.

## Planner-only retry after checksum metadata fix

Keep rootcause-001 intact. No environment runs are repeated:

```bash
(
set -e
PYTHONPATH=src python -m unittest discover -s tests -v
timeout --signal=TERM --kill-after=15s 960s \
  python scripts/phase1_root_cause.py \
  --source results/phase1-push-collectorcheck-003 \
  --planner-only \
  --out results/phase1-push-rootcause-planner-002
)
```

Maximum 250 planner calls, zero environment transitions, zero optimizer updates.
Share the new summary.json; for worker errors share the corresponding worker log.
Previous A1 evidence stays separate, not relabeled as newly measured.
