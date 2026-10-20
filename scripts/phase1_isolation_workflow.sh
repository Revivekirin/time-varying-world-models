#!/usr/bin/env bash
# One bounded gate sequence, one audited continuation; no retries or new weights.
set -euo pipefail
cd "$(dirname "$0")/.."
tag=${TVWM_RUN_TAG:-isolation-001}
[[ "$tag" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 1
workflow=results/phase1-workflow-${tag}
push_gate=results/phase1-push-collectorcheck-${tag}
pick_gate=results/phase1-pick-collectorcheck-${tag}
resume_out=results/phase1-push-resume-${tag}
for destination in "$workflow" "$push_gate" "$pick_gate" "$resume_out"; do
  if [[ -e "$destination" ]]; then echo "Refusing existing output: $destination" >&2; exit 1; fi
done
mkdir -p "$workflow"
python scripts/run_server_tests.py 2>&1 | tee "$workflow/unit-tests.log"
timeout --signal=TERM --kill-after=15s 600s python scripts/phase1_collector_check.py \
 --preflight results/phase1-push-preflight-002 --resume-check results/phase1-push-resumecheck-001 \
 --out "$push_gate" 2>&1 | tee "$workflow/push-collector.log"
timeout --signal=TERM --kill-after=15s 600s python scripts/phase1_collector_check.py \
 --preflight results/phase1-pick-preflight-001 --resume-check results/phase1-pick-resumecheck-001 \
 --out "$pick_gate" 2>&1 | tee "$workflow/pick-collector.log"
timeout --signal=TERM --kill-after=15s 210s python scripts/phase1_isolation_resume.py \
 --source results/phase1-push-seed1-wandb-001 --push-gate "$push_gate" --pick-gate "$pick_gate" \
 --out "$resume_out" 2>&1 | tee "$workflow/checkpoint-audit.log"
# The runner retains the original cumulative wall ledger, including crash downtime.
timeout --signal=TERM --kill-after=120s 86400s python scripts/phase1_train.py \
 --preflight results/phase1-push-preflight-002 --resume-check results/phase1-push-resumecheck-001 \
 --collector-check "$push_gate" --approve-budget 300000 --resume --acknowledge-rollback \
 --out "$resume_out" 2>&1 | tee "$workflow/push-training.log"
