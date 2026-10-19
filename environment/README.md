# Installation gates

The user exclusively executes remote commands. No SSH automation is provided.
Do not install the reference requirements blindly. First collect host evidence:

```bash
cd /path/to/time-varying-world-models
python3 scripts/inventory.py reports/runtime/server-before
```

Return `inventory.json`. This reads OS/architecture, Python, installed packages,
CUDA runtime (if torch exists), driver/GPU, native shared libraries and Vulkan.
No sudo, downloads, simulation, or package changes occur. A new output directory
is required on every invocation. Do not print credential files.

**UNRESOLVED:** Python/CUDA wheel channel, driver compatibility, SAPIEN wheel,
Vulkan ICD availability, and a complete transitive dependency lock. The macOS
Python 3.14 environment is not a substitute for Linux CUDA verification.

After inventory review, select a supported isolated Python (upstream references
3.11), CUDA PyTorch build, and resolve this minimal set plus the pinned ManiSkill
checkout. Do not install original TD-MPC, gym 0.21, ManiSkill2, MuJoCo, W&B, or the
entire upstream environment for this adapter. Upstream `common` imports required
by the runner use torch, torchrl, tensordict, Hydra and OmegaConf.

The following is a **conditional procedure, not a command to execute before review**:

```bash
python3.11 -m venv .venv-gpu
source .venv-gpu/bin/activate
python scripts/fetch_sources.py
# Resolve reviewed package versions/channel into environment/requirements-linux.lock.
# A requirements-linux.lock does not yet exist: do not substitute the reference file.
python -m pip install -r environment/requirements-linux.lock
python -m pip install --no-deps -e external/ManiSkill -e .
python -m pip check
python -m pip freeze --all > reports/runtime/installed-freeze.txt
python scripts/inventory.py reports/runtime/server-installed
PYTHONPATH=src python -m unittest discover -s tests -v
```

The reviewed lock must pin all resolved transitive packages and CUDA package
variants; record wheel hashes/index, Python version and source lock alongside it.
Build the lock using a dry-run resolver report on the actual server before
installation, then verify `pip check`. A freeze is a post-install snapshot, not
by itself a portable hashed lock. Record external native dependencies separately.
If resolution fails, retain the resolver error and stop; do not change system
packages. If a genuine framework conflict remains, first assess minimal external
compatibility changes, then isolated processes with explicitly tested tensor and
transition serialization. No such conflict has yet been executed/established.

After API/unit tests pass, follow `configs/experiment/phase0_manifest.json` and
`scripts/REMOTE_COMMANDS.md`. Stop on any failed gate; send the failed report.
