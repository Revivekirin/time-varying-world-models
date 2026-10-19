# Phase 1 go/no-go

As of the user-supplied fresh-process learner reports: **NO-GO until the new
collector/evaluation check passes and the user explicitly approves the full budget.**
No full training has been executed by this agent.

| Gate | Status |
|---|---|
| Phase 0 contracts, replay, model/planner/learner and frozen evaluation | VERIFIED by supplied reports and preflight |
| Batch 256 forward/backward/optimizer, full replay storage | VERIFIED on both tasks |
| Active GPU headroom and available disk at preflight time | VERIFIED; rechecked at launch and during training |
| Official warmup/update cadence and episodic handling | VERIFIED against pinned source |
| Same-process and fresh-process learner restore | VERIFIED on both tasks |
| Shared collector episode-boundary continuation in a fresh process | UNRESOLVED pending phase1_collector_check.py |
| Full runner frozen evaluator compatibility | UNRESOLVED; included in that check with two seeds |
| Full runner CUDA end-to-end execution | UNRESOLVED; implementation supplied, local GPU unavailable |
| Explicit 300,000-transition/task budget approval | Required via user-executed --approve-budget 300000 |
| Nominal competence and stability | UNRESOLVED; no full-run results yet |

The runner checks task identity, evidence statuses, checkpoint hashes, exact
source/package/lock provenance, code/config fingerprint, actual environment
contract, free memory and storage before training. Any affected failed gate stops
that task. Run the two bounded checks first; do not combine them with an automatic
full-training launch. The explicit launch command is a separate user decision.

No 10,000-transition pilot, dynamics shifts, history inputs, online adaptation,
changed objectives, changed planning hyperparameters, or automatic Phase 2.
