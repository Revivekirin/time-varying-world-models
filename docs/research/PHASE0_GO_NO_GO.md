# Phase 0 go/no-go

**PHASE 0 STATUS: PARTIAL**
**Phase 1 nominal TD-MPC2 training: NOT READY for separate launch approval.**

VERIFIED: exact official source pins and audit; local dependency-free contract
tests pass. External adapter, shared agent bridge, bounded runners and future
research plans are implemented/prepared. No upstream source patch or shift code.

UNRESOLVED for BOTH PushCube-v1 and PickCube-v1: installed dependency compatibility,
actual task contracts/dimensions, simulator execution, model/planner execution,
TorchRL replay sampling, finite learner updates, checkpoint execution, GPU/runtime
costs. Neither task has passed runtime readiness. No competent nominal checkpoint.

Next: user runs inventory; review actual Linux driver/Python/native libraries,
resolve a reproducible dependency lock and install in isolation. User then executes
unit/API gates, contract smoke, learner smoke and frozen evaluation in order.
Collect the six task reports and GPU telemetry. A failed task remains task-specific
partial completion; do not hide it behind another task's successful smoke.

Additional open constraints: multi-env GPU behavior is source-inspected only;
exact state/controller/RNG/time restoration not established; checkpoint is not
training-resumable. A pass for bounded single-env integration would not establish
these broader capabilities, nominal competence, interventions or history benefit.

## Latest gate update

Server installation/import/CUDA/unit gates and both single-env A/B contract
gates now PASS by user execution. Earlier unresolved installation and task-shape
entries are superseded by PHASE0_TEST_RESULTS.md updates. Model/planner, replay
sampler, learner, checkpoint and full memory/runtime gates remain unresolved.
Phase0 PARTIAL; Phase1 not ready.

## Learner gates update

Both tasks now pass model/planner, actual replay sampling,8-update learner smoke,
Polyak target check and checkpoint/frozen-inference tests. Remaining next gates:
independent frozen rollout evaluation and device-wide memory evidence. No competent
nominal policy established. See latest PHASE0_TEST_RESULTS and archived reports.
