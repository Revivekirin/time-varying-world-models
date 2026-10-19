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
