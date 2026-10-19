# User-run Phase 1 commands

First sync the new implementation to the server using your normal Git workflow.
Never overwrite old outputs. Use new suffixes if these paths already exist.
No remote execution is performed by the agent.

## 1. Bounded final compatibility check (not a training pilot)

Maximum 250 transitions, zero optimizer updates per task. Three collector episodes
(reference/fresh-process comparison) plus two frozen evaluation episodes.

```bash
conda activate tvwm-phase0
cd /nfs4/jhkim/repos/time-varying-world-models
(
set -e
PYTHONPATH=src python -m unittest discover -s tests -v
timeout --signal=TERM --kill-after=15s 600s \
  python scripts/phase1_collector_check.py \
  --preflight results/phase1-push-preflight-002 \
  --resume-check results/phase1-push-resumecheck-001 \
  --out results/phase1-push-collectorcheck-002
timeout --signal=TERM --kill-after=15s 600s \
  python scripts/phase1_collector_check.py \
  --preflight results/phase1-pick-preflight-001 \
  --resume-check results/phase1-pick-resumecheck-001 \
  --out results/phase1-pick-collectorcheck-001
)
```

Require both PASS_COLLECTOR_BOUNDARY and all server unit tests to pass, with no
unexpected dependency skips. Inspect the reports before deciding to launch.
Failures stop the affected task. Do not lower batch size or change architecture.
Code/config edits after this check invalidate the gate and require rerunning it.

## 2. Explicitly approved full run (separate user action)

Running the following commands with --approve-budget 300000 explicitly approves
300,000 training transitions / 299,999 learner updates per task, seed 1, model 5M,
batch 256, up to 31,000 validation transitions, 24 h/task, 20 GiB used-memory
ceiling, 4 GiB free GPU and 30 GiB free storage floors. Tasks execute sequentially.
The runner independently checks preflight/resume/collector gates and provenance.
This document is a prepared command, not evidence that approval or training occurred.

```bash
conda activate tvwm-phase0
cd /nfs4/jhkim/repos/time-varying-world-models
(
set -e
timeout --signal=TERM --kill-after=120s 86400s \
  python scripts/phase1_train.py \
  --preflight results/phase1-push-preflight-002 \
  --resume-check results/phase1-push-resumecheck-001 \
  --collector-check results/phase1-push-collectorcheck-002 \
  --approve-budget 300000 \
  --out results/phase1-push-seed1-001
timeout --signal=TERM --kill-after=120s 86400s \
  python scripts/phase1_train.py \
  --preflight results/phase1-pick-preflight-001 \
  --resume-check results/phase1-pick-resumecheck-001 \
  --collector-check results/phase1-pick-collectorcheck-001 \
  --approve-budget 300000 \
  --out results/phase1-pick-seed1-001
)
```

No automatic relaunch on failure. Progress is printed every 1,000 transitions;
metrics-*.jsonl and gpu-*.csv update during the run. Initial validation precedes
collection. Early random collection has no optimizer updates; step 1000 includes
the official 1,000-update burst. The runner retains latest, interval, best and
final artifacts. COMPLETE_NOMINAL_TRAINING is budget completion, not competence.

## 3. Interrupted-run resume, only after diagnosing the interruption

Example for Push; use corresponding Pick paths for Pick. Do not change output
path on resume. The explicit rollback acknowledgement accepts repeated physical
work since the last durable boundary, which is recorded in resume-*.json.
Cumulative wall time persists; exceeding it requires a new reviewed budget,
not an automatic retry. Use only latest.pt with its matching SHA sidecar.

```bash
timeout --signal=TERM --kill-after=120s 86400s \
  python scripts/phase1_train.py \
  --preflight results/phase1-push-preflight-002 \
  --resume-check results/phase1-push-resumecheck-001 \
  --collector-check results/phase1-push-collectorcheck-002 \
  --approve-budget 300000 --resume --acknowledge-rollback \
  --out results/phase1-push-seed1-001
```

## 4. Final heldout evaluation after selection

Separately run after both tasks complete. Up to 5,000 transitions/task, no updates,
100 independent seeds; no checkpoint reselection. This is nominal evaluation,
not Phase 2. Reports include per-seed outcomes and selected model hashes.

```bash
(
set -e
timeout --signal=TERM --kill-after=15s 900s \
  python scripts/phase1_train.py --heldout \
  --out results/phase1-push-seed1-001 \
  --eval-output results/phase1-push-seed1-001/heldout.json
timeout --signal=TERM --kill-after=15s 900s \
  python scripts/phase1_train.py --heldout \
  --out results/phase1-pick-seed1-001 \
  --eval-output results/phase1-pick-seed1-001/heldout.json
)
```
