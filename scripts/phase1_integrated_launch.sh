#!/usr/bin/env bash
# Explicitly authorized integrated gates -> sequential nominal runs. No retries.
set -euo pipefail
cd "$(dirname "$0")/.."
run_tag=${TVWM_RUN_TAG:-recreate-001}
if [[ ! "$run_tag" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo 'TVWM_RUN_TAG must contain only letters, digits, underscores or hyphens.' >&2
  exit 1
fi
workflow=results/phase1-workflow-${run_tag}
push_gate=results/phase1-push-collectorcheck-${run_tag}
pick_gate=results/phase1-pick-collectorcheck-${run_tag}
push_run=results/phase1-push-seed1-${run_tag}
pick_run=results/phase1-pick-seed1-${run_tag}
for destination in "$workflow" "$push_gate" "$pick_gate" "$push_run" "$pick_run"; do
  if [[ -e "$destination" ]]; then
    echo "Refusing existing output: $destination. No automatic retry or overwrite." >&2
    exit 1
  fi
done
mkdir -p "$workflow"
stage=unit-tests
trap 'echo "Stopped at $stage. No automatic retry. Inspect logs under $workflow." >&2' ERR
# A skipped GPU/dependency test does not qualify as successful server verification.
PYTHONPATH=src python - <<'PY' 2>&1 | tee "$workflow/unit-tests.log"
import unittest
suite=unittest.defaultTestLoader.discover('tests')
result=unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful() or result.skipped:
    raise SystemExit('Integrated server gate requires all tests passing without skips')
PY
stage=push-collector
timeout --signal=TERM --kill-after=15s 600s \
  python scripts/phase1_collector_check.py \
  --preflight results/phase1-push-preflight-002 \
  --resume-check results/phase1-push-resumecheck-001 \
  --out "$push_gate" 2>&1 | tee "$workflow/push-collector.log"
stage=pick-collector
timeout --signal=TERM --kill-after=15s 600s \
  python scripts/phase1_collector_check.py \
  --preflight results/phase1-pick-preflight-001 \
  --resume-check results/phase1-pick-resumecheck-001 \
  --out "$pick_gate" 2>&1 | tee "$workflow/pick-collector.log"
# Both gates completed before either full run. Each runner independently verifies
# gate identity, lifecycle, code hashes, preflight, dependency and resource limits.
stage=push-training
timeout --signal=TERM --kill-after=120s 86400s \
  python scripts/phase1_train.py \
  --preflight results/phase1-push-preflight-002 \
  --resume-check results/phase1-push-resumecheck-001 \
  --collector-check "$push_gate" \
  --approve-budget 300000 --out "$push_run" 2>&1 | tee "$workflow/push-training.log"
stage=pick-training
timeout --signal=TERM --kill-after=120s 86400s \
  python scripts/phase1_train.py \
  --preflight results/phase1-pick-preflight-001 \
  --resume-check results/phase1-pick-resumecheck-001 \
  --collector-check "$pick_gate" \
  --approve-budget 300000 --out "$pick_run" 2>&1 | tee "$workflow/pick-training.log"
echo 'Both nominal training budgets completed. Inspect validation results; competence and Phase 2 are not implied.'
