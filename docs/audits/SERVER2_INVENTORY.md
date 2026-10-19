# server2 inventory review

User-provided runtime evidence, 2026-10-19 17:35:41. No agent SSH execution.

VERIFIED by supplied inventory: Linux kernel5.15.0-186, x86_64, glibc2.35;
Conda tvwm Python3.12.0; all nine queried research packages absent;
RTX3090 24576MiB; NVIDIA535.171.04, nvidia-smi CUDA display12.2;
1MiB occupied and no GPU processes at inventory time. libcuda, libvulkan,
NVIDIA GLX/EGL present in ldconfig. vulkaninfo command unavailable.
The missing diagnostic executable does not prove Vulkan is broken; NVIDIA ICD
selection and SAPIEN GPU runtime remain UNRESOLVED.

Decision: separate tvwm-phase0 Python3.11 environment, preserving existing tvwm.
Retain upstream NumPy1.24.4 and algorithm dependency versions. Candidate
Torch2.7.1+cu118 is officially distributed and uses older CUDA runtime supported
by driver backward compatibility. Do not install CUDA toolkit or change driver.
This choice does not establish SAPIEN/Torch interop; execute runtime gates later.
SAPIEN3.0.3 candidate matches inspected ManiSkill minimum. Resolve full dependencies
with pinned editable ManiSkill source before installing research packages.

Next user-run step: conda create isolated Python3.11/pip; fetch pinned sources;
pip install --dry-run --report using requirements-server2.in plus editable
ManiSkill, PyPI primary index and official PyTorch cu118 extra index.
A dry run may download wheels/build metadata in temporary isolation but does not
install requested research packages into the target environment. Save exact report
and error/log; do not treat the candidate input as the transitive lock.

After resolution, review selected URLs/versions/hashes, generate installation lock,
install, verify pip check and CUDA tensor operation, then SAPIEN/task contracts.
No simulation, learner updates or Phase1/2 execution authorized by this review.

Sources:
- https://pytorch.org/get-started/previous-versions/
- https://pypi.org/project/numpy/1.24.4/
- https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
- https://pypi.org/project/sapien/3.0.3/

## Resolver summary received

User supplied complete name/version/URL summary of
reports/runtime/resolve-20261019-174022/resolve.json. Python3.11.17;
101 selected entries including editable ManiSkill. Major candidates match,
Torch is +cu118; Linux wheels require at most glibc2.28 in this selection,
below observed2.35. The original report hashes remain on the server, not in
this summary. environment/server2-resolved-summary.json preserves supplied data.
Successful resolution is not installed binary/API compatibility evidence.

scripts/lock_from_report.py converts the original report into a hash-checked,
exact-URL artifact lock, validates major versions and permitted artifact origins,
and excludes the pinned editable ManiSkill checkout. Refuses overwrite.
Local synthetic conversion test passed (100 artifact entries) and missing-hash
rejection passed; no actual remote hashes validated locally.
ANTLR4 runtime is a source archive; its isolated build dependencies are not
covered by this artifact lock. Preserve pip installation report/build log and
later retain built wheels for stronger reproducibility. Editable project build
tooling is likewise separate. Do not call this a fully hermetic environment.

Next user execution: verify source pins; generate lock; pip install locked
artifacts; install editable projects without dependency resolution; pip check;
CUDA tensor/import checks; all unit tests. No simulation until these pass.

## Installation attempt blocked at lock generation

User log: initial script absent, then retrieved with git pull. Source verification
passed for all three commits. Lock conversion failed on jinja2 with the old
combined missing-hash/invalid-URL check. The pasted log does not distinguish
missing archive SHA256 from an unsupported hash representation or fragment.
No locked dependency install occurred. Editable mani_skill/tvwm installed with
--no-deps; pip check listed missing dependencies, and torch import failed.
User-run tests: 8 pass, 2 tensor tests skipped. No CUDA/simulation evidence.

Fixed converter accepts modern and legacy SHA256 fields and SHA256 URL fragments,
rejects malformed/conflicting hashes, and offers explicit --hash-missing to
fingerprint exact HTTPS artifacts lacking SHA256. Such fingerprints are labeled
as downloaded bytes, not publisher checksums. Original report is preserved.
Local regression tests: 7 converter tests pass, total15 pass/2 tensor skips.
Download path tested with a mock, not a remote installation. Retry commands must
stop on failure; do not continue to editable install/import checks after errors.

## Installation/import gate subsequently passed

User log confirms locked dependency installation, clean pip check,
Torch2.7.1+cu118/CUDA11.8 on RTX3090, successful CUDA tensor operation and
all requested imports, and17/17 unit tests passing in2.257s with no skips.
This supersedes the previous installation failure. Runtime simulator/planner/
learner/checkpoint behavior remains untested. Next: bounded contract tests only.
