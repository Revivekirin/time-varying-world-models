# Phase 1 go/no-go

**FULL TRAINING: NO-GO pending mandatory preflight and final budget approval.**
The previous10000-transition pilot is canceled by user instruction. Proposed full
budget is300000 transitions/task, seed1, model5M, batch256, env1; this document
does not treat a proposal as explicit launch approval.

| Gate | Status |
|---|---|
| Phase0 source, task contracts, model/planner/replay/learner smoke | VERIFIED by supplied reports |
| Model-only checkpoint/frozen inference | VERIFIED |
| Independent evaluation episode records | UNRESOLVED locally; mandatory preflight input |
| Batch256 forward/backward/optimizer, full replay allocation | UNRESOLVED; prepared command |
| Active GPU memory/free memory and storage | UNRESOLVED; automatically measured by preflight |
| Upstream cadence/episodic source audit | VERIFIED source; exact budget endpoint documented |
| Learner resume envelope | IMPLEMENTED, runtime UNRESOLVED |
| Collector and fresh-process resume | UNRESOLVED; not launch-ready |
| Full runner, wall deadline, retention and metrics implementation | Pending preflight results; not launch-ready |
| Explicit final full-budget approval | Pending |

Run only scripts/phase1_preflight.py using configs/phase1/preflight_manifest.json.
PASS_COMPATIBILITY_ONLY does not authorize full training and does not establish
collector resume. Stop any affected task on failure. Keep outputs separate from
all Phase0 results. After compatibility evidence, complete fresh-process collector
resume verification and present final launch command/resource budget for approval.

No checkpoints or results are invented. Nominal competence requires validation
and heldout seed results, goal progress and stability; declining losses are not
enough. One training seed cannot establish across-training-seed reproducibility.
If nominal learning fails, inspect integration/reward/replay/optimization. No
Phase2, hidden dynamics, history conditioning or online adaptation is initiated.
