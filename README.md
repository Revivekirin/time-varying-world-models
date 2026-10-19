# Time-Varying World Models

Phase 0 integration of **official TD-MPC2 × ManiSkill3**, state observations,
PushCube-v1 and PickCube-v1. **PHASE 0 STATUS: PARTIAL. Phase 1 is not ready.**

Source audit, an external adapter, bounded smoke commands, and local contract
unit tests are provided. No GPU simulation or learner update has been executed.
The user alone runs remote commands. The next action is host inventory:

```bash
python3 scripts/inventory.py reports/runtime/server-before
```

Return the generated `inventory.json` before dependency selection/install.
See [installation gates](environment/README.md), [remote commands](scripts/REMOTE_COMMANDS.md),
[bounded manifest](configs/experiment/phase0_manifest.json), and
[test evidence](docs/audits/PHASE0_TEST_RESULTS.md).

Local checks (no research dependencies):

```bash
python3 -m venv .venv-local
PYTHONPATH=src .venv-local/bin/python -m unittest discover -s tests -v
.venv-local/bin/python -m compileall -q src scripts tests
```

[Source pins](environment/sources.lock.json) reproduce unmodified checkouts using
`python3 scripts/fetch_sources.py`. `external/` is intentionally ignored: upstream
source, licenses, media and datasets are not copied into this repository. Exact
commit and clean-tree checks run before GPU tests. No source patch is applied.
Use `git diff` within any checkout to inspect a suspected modification.

`src/tvwm/envs` handles observation/action/episode boundaries; `agents` imports
official model/planner/replay code; `scripts` contains bounded runners.
`shifts` and `identification` are empty scope placeholders. `results/` contains
isolated, non-overwriting runtime outputs and is ignored. Reports in `docs/audits`
record evidence; large checkpoints/recordings are excluded from Git.

Research scope and gates:
- [Research sequence and controls](docs/research/RESEARCH_SCOPE.md)
- [Upstream audit](docs/audits/UPSTREAM_AUDIT.md)
- [Dependency compatibility](docs/audits/DEPENDENCY_COMPATIBILITY.md)
- [PushCube](docs/contracts/PUSHCUBE_CONTRACT.md), [PickCube](docs/contracts/PICKCUBE_CONTRACT.md)
- [TD-MPC2 integration](docs/contracts/TDMPC2_INTEGRATION.md)
- [Future intervention audit](docs/audits/DYNAMICS_INTERVENTION_FEASIBILITY.md)
- [Phase 1 proposal](docs/research/PHASE1_NOMINAL_TRAINING_PLAN.md)
- [Go/no-go](docs/research/PHASE0_GO_NO_GO.md)

VERIFIED means inspected source or actual executed evidence, explicitly labeled
which. INFERRED means reasoned interpretation. UNRESOLVED means not established.
A smoke success would show pipeline operation only, never nominal competence.
