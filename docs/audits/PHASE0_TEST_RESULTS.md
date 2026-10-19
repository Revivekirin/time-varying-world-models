# Phase 0 executed evidence

**PHASE 0 STATUS: PARTIAL** — 2026-10-19. This is not a runtime readiness pass.

## VERIFIED local execution

Host: Darwin 27.0.0 arm64, Python 3.14.7; isolated `.venv-local`.
`reports/runtime/local-before/inventory.json` holds local read-only inventory.
No Linux server inspection or remote connection was performed; user owns remote execution.

Commands executed from repository root:

```bash
uname -a
python3 --version
git status --short
git clone https://github.com/nicklashansen/tdmpc2.git external/tdmpc2
git clone https://github.com/nicklashansen/tdmpc.git external/tdmpc
git clone https://github.com/haosulab/ManiSkill.git external/ManiSkill
python3 -m venv .venv-local
.venv-local/bin/python scripts/inventory.py reports/runtime/local-before
.venv-local/bin/python scripts/fetch_sources.py
PYTHONPATH=src .venv-local/bin/python -m unittest discover -s tests -v
.venv-local/bin/python -m compileall -q src scripts tests
git diff --check
```

The first sandboxed clone failed DNS resolution; the permitted network retry
succeeded. All source commits and clean trees were then verified. Source audit
used `rg`, `cat`, and `sed` on files named in UPSTREAM_AUDIT and task contracts.
Existing project .git was empty (no commits); no prior project code was modified.
No source checkout modifications, dependency upgrades or large artifacts added to Git.

Unit test execution: 10 discovered, **8 passed, 2 skipped**. Stdlib tests verify
asymmetric action roundtrip, invalid action/bounds, replay indexing, truncation
bootstrap, simultaneous flags, cross-episode rejection, incomplete episode rejection,
and shape/length rejection. Skips are tensor observation conversion and TensorDict
packing because torch/tensordict are not installed in the local environment.
`unittest` displayed 0.000s at its reporting resolution; this is not throughput data.
Compile check passed. Source checkout pin verification passed. These checks do not
execute model, tensor adapter, planner, simulator or optimizer.

## UNRESOLVED target runtime

| Test | PushCube-v1 | PickCube-v1 |
|---|---|---|
| A construction/reset/step/boundary | NOT RUN | NOT RUN |
| B finite prescribed-action behavior | NOT RUN | NOT RUN |
| C model/policy/planner forward | NOT RUN | NOT RUN |
| D real upstream sequence sampling | NOT RUN | NOT RUN |
| E finite losses/gradients | NOT RUN | NOT RUN |
| F fixed learner updates | NOT RUN | NOT RUN |
| G collection/training integration | NOT RUN | NOT RUN |
| H checkpoint/frozen inference/evaluation | NOT RUN | NOT RUN |
| I GPU memory/throughput/latency | NOT RUN | NOT RUN |

**Executed environment transitions: 0. Executed physics steps: 0.
Executed learner updates: 0. GPU time/memory: unmeasured. Simulation/planner/learner
wall times: unmeasured. No research checkpoints produced.**

Source-based observation dimensions (35/42) are INFERRED, not measured.
Target environment versions, exact native library availability and dependency
lock remain UNRESOLVED pending server inventory. Missing assets, dependency
conflicts or GPU setup errors must stop the affected gate and be reported exactly.
No fabricated conflict or successful installation is asserted.

Prepared manifest: configs/experiment/phase0_manifest.json, with exact commands,
resets, transitions, updates, artifacts, timeouts and stop conditions. Per task:
contract <=100 transitions/0 updates, learner <=250/8, independent eval <=100/0.
Across both tasks <=900 transitions, <=16 updates, and <=4500 physics substeps
at the audited frequencies, excluding any internal reset initialization physics.
Synthetic replay fixtures are not simulator transitions. Native constructors may
reset once in addition to explicitly counted resets.

All GPU commands are **prepared and unexecuted**. After successful execution,
archive the report.json/config/installed versions/source pins and device-wide
memory telemetry. A report PASS is a bounded-stage result, not global Phase 0
completion or evidence of a competent robot policy. Do not expand failing tests.

## Update: user-executed server dependency gate passed

User supplied server2 installation/test log after lock-generator fix:
- Hashed artifact installation completed; editable projects already installed.
- `pip check`: No broken requirements found.
- Torch2.7.1+cu118, runtime CUDA11.8, NVIDIA GeForce RTX3090.
- CUDA tensor arithmetic and imports (SAPIEN, ManiSkill environments, TensorDict,
  TorchRL) passed.
- All17 unit tests passed, no skips, reported duration2.257s. Includes real CPU
  tensor observation conversion and TensorDict packing, not GPU replay sampling.

VERIFIED by user execution, not agent remote execution. The previous missing-
package blocker is resolved. Full lock exists on the server; its contents have
not yet been copied to this local repository. Simulation transitions and learner
updates remain0 in supplied evidence. CUDA test is not a simulator test.
Next gate: per-task bounded contract A/B (<=100 transitions/task,0 updates).
Global Phase0 remains PARTIAL; Phase1 not ready.

## First PushCube contract attempt

User report: FAIL before explicit reset/step,0 transitions,0 updates,5.209s.
Guard rejected combined horizon/frequencies; prior error omitted actual values,
so the failed field is not proven by this log. PickCube command did not run.
Source inspection found an adapter defect: ManiSkill TimeLimitWrapper replaces
Gymnasium TimeLimit, so env.spec.max_episode_steps need not retain the horizon.
The pinned upstream provides find_max_episode_steps_value for this exact case.
Adapter now uses that utility, records horizon provenance/spec value, and reports
actual horizon and frequencies on any mismatch. Expected50/100Hz/20Hz and all
transition budgets are unchanged. No upstream patch or simulator setting change.
Added simulator-free installed-framework regression test for a wrapper limit with
missing spec limit; remote execution pending. Retry in a fresh output directory;
retain push-contract-001 as failed evidence. Global status remains PARTIAL.
