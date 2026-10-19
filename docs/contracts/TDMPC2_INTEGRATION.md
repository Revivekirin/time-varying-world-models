# TD-MPC2 integration contract

Status: implementation prepared, GPU runtime UNRESOLVED. No algorithm fork/patch.
One shared agent/environment bridge serves both task configs. Independent processes
avoid upstream global imports and shared encoder dictionary across agents.

`load_api` verifies exact clean commits before importing official TDMPC2 and Buffer.
Configuration starts at upstream config.yaml and MODEL_SIZE[5], records all resolved
fields. episodic=true, compile=false, batch_size=16, capacity/steps=4096 are explicit
smoke overrides. Compile off is diagnostic; objective/planner/reward/discount
coefficients remain upstream. Upstream default batch 256 resumes in Phase 1 proposal.
No upstream train.py invocation: its seed-update burst/inclusive loop would exceed
our bounded budget. Collection schedule changes are runner-only smoke controls.

Observation/action: native state only, float32; no normalization of observations,
reward rescaling, history, privileged context or imagery. Native normalized action
space is identity to planner [-1,1]; ManiSkill performs physical-unit scaling once.
Generic affine helper is tested for asymmetric bounds but is not applied twice.

Replay: rows are (o0, NaN, NaN, NaN), (o1,a0,r0,terminated0), ... . Complete episodes
only. Scalar truncation closes the trajectory but is not stored as termination.
Both flags true closes and suppresses bootstrap because terminated=true. Native
final observation is inserted before reset; no autoreset wrapper. Sequence length
horizon+1 and episode IDs are delegated to unmodified upstream Buffer/SliceSampler.
A synthetic sampler test checks observation increments/action/reward alignment,
no cross-episode sampling, and sentinel exclusion. No agent training on synthetic
transitions. Episodes shorter than horizon stop smoke rather than unbounded retries.

Tests cover encoder, latent dynamics, reward/Q/termination outputs, policy prior,
MPPI bounds, actual replay sampling, finite losses and gradient norms, <=8 learner
updates, target Polyak behavior, save/load, seeded frozen inference, and independent
frozen evaluation. See exact limits in phase0_manifest.json. Model computations,
soft CE, consistency/BCE, entropy scaling, target networks and Adam remain upstream.

Checkpoint: upstream model-only agent.pt is for evaluation. RNG snapshot records
Python/NumPy/torch CPU/CUDA states; resolved_config and source hashes accompany it.
It is NOT an exact training-resume checkpoint: no optimizer, replay, scale or
simulator/RNG-time snapshot restore. Phase 1 must either add a separately tested
resume envelope or explicitly prohibit resume and count interrupted runs as failed.
Frozen evaluation disables gradients and updates, uses eval_mode=True and held-out
seeds. MPPI still samples; equal seed/RNG tests are needed for matching outputs.

Single environment is deliberate. Batched simulation exists, but scaling upstream
single-cache planner to N>1 without per-environment state would mix episodes.
No vectorized learner/planner throughput claim is made. Source state restoration
APIs do not guarantee exact future rollout replay. Both limitations remain explicit.
