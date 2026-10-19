# Dependency compatibility

VERIFIED local host: Darwin 27.0.0 arm64, Python 3.14.7. An isolated `.venv-local`
was created for stdlib unit tests. No research framework was installed locally.
Remote Linux, RTX3090 24GB is the requested target, not inspected hardware.

| Dependency | Inspected upstream constraint | Actual target installed version |
|---|---|---|
| Python | TD-MPC2 environment: 3.11 | UNRESOLVED |
| Torch | 2.7.1; Docker CUDA12.6/cuDNN9 | UNRESOLVED |
| Gymnasium | TD-MPC2 .29.1; ManiSkill >=.29.1 | UNRESOLVED |
| NumPy | TD-MPC2 1.24.4; ManiSkill >=1.22 | UNRESOLVED |
| TensorDict/TorchRL | .8.3 / .8.1 | UNRESOLVED |
| Hydra/OmegaConf | 1.3.2 / 2.3.0 | UNRESOLVED |
| ManiSkill | pinned source setup version 3.0.1 | UNRESOLVED |
| SAPIEN | ManiSkill Linux >=3.0.3 | UNRESOLVED |
| mplib / pytorch_kinematics | .1.1 Linux / .7.6 | UNRESOLVED |
| Driver / Vulkan / native libraries | inventory required | UNRESOLVED |

VERIFIED source constraints overlap for Gymnasium and NumPy; this is not a
successful resolver result. ManiSkill also requests scipy, h5py, transforms3d,
trimesh, imageio, PyYAML, dacite, GitPython, tabulate, IPython, defusedxml,
nvidia-ml-py, tyro>=.8.5 and huggingface_hub. Resolve all transitives, do not let
unreviewed latest packages silently redefine the experiment.

Known interface incompatibility: old ManiSkill2 wrapper uses the old environment
ID/four-return convention/action repeat. Use external ManiSkill3 adapter. Do not
force gym0.21 or mani-skill2 .4.1 into this environment. Original TD-MPC's old
stack is not an integration dependency. No actual pip conflict is claimed yet.

TD-MPC2 hard-codes cuda:0, uses torch.nn.Buffer, capturable Adam, TensorDictParams
and torch.vmap. CUDA smoke cannot be verified with an MPS/CPU replacement.
SAPIEN uses Vulkan/native physics; torch CUDA availability alone is insufficient.
No sudo or system-package upgrades were attempted. Missing native dependencies
are a blocker to report rather than an invitation to modify the server.

Source lock is complete. **Resolved binary/transitive dependency lock is not yet
available**, deliberately pending server inspection. requirements.reference.txt
is audit evidence only, not an install lock. Follow environment/README.md for
inventory, reviewed resolution, isolated installation and post-install recording.

## Update: server2 inventory received

See SERVER2_INVENTORY.md for user-executed evidence and reviewed candidate.
The earlier remote-inventory blocker is resolved. Actual research package
installation, resolver lock and runtime compatibility are still unresolved.
Python3.11 in a separate environment and Torch2.7.1+cu118 are selected for the
next resolver dry run; requirements-server2.in is not a completed lock.
