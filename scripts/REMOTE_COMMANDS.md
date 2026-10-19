# User-run remote commands

Run only inventory now (no GPU simulation):

```bash
cd /path/to/time-varying-world-models
python3 scripts/inventory.py reports/runtime/server-before
```

Send `reports/runtime/server-before/inventory.json` for dependency selection.
Use a different output suffix if a directory already exists. Install only after
that review, using `environment/README.md`. GPU tests below are prepared, not run.

After installation, source pins, `pip check`, and unit tests pass:

```bash
python scripts/fetch_sources.py
PYTHONPATH=src python -m unittest discover -s tests -v
 timeout --signal=TERM --kill-after=15s 330s python scripts/smoke.py --task PushCube-v1 --stage contract --out results/push-contract-001
 timeout --signal=TERM --kill-after=15s 330s python scripts/smoke.py --task PickCube-v1 --stage contract --out results/pick-contract-001
```

Review both contract reports and stop for any failure, unexpected dimensions,
controller mapping, nonfinite value, missing boundary, or seed mismatch. Then:

```bash
 timeout --signal=TERM --kill-after=15s 930s python scripts/smoke.py --task PushCube-v1 --stage learner --contract results/push-contract-001/report.json --out results/push-learner-001
 timeout --signal=TERM --kill-after=15s 930s python scripts/smoke.py --task PickCube-v1 --stage learner --contract results/pick-contract-001/report.json --out results/pick-learner-001
```

Only after successful learner reports:

```bash
 timeout --signal=TERM --kill-after=15s 930s python scripts/evaluate.py --smoke results/push-learner-001 --out results/push-eval-001
 timeout --signal=TERM --kill-after=15s 930s python scripts/evaluate.py --smoke results/pick-learner-001 --out results/pick-eval-001
```

For device-wide memory (PyTorch counters exclude SAPIEN), in another terminal
start `nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 1`
and save its output for the duration of each test; stop it when the test finishes.
Record other GPU processes and baseline memory. Share report.json, resolved_config.json,
installed inventory/freeze and GPU telemetry; model binaries need not be pasted.
No Phase 1 training or Phase 2 rollouts are included.
