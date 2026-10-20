# Collector boundary root-cause investigation

**Readiness: NO-GO. Root cause: UNRESOLVED. No training or diagnostic GPU execution performed locally.**

## VERIFIED user-provided evidence

32 server tests pass. Initial observation and first action match exactly. First reported observation divergence occurs after action 0: obs[1] max difference 4.48785722e-05, exceeding atol=1e-6. Action[1] difference 2.98023224e-08 is nonzero but within tolerance. Action[2] difference 0.576643586 is outside tolerance. Counters match; no optimizer updates.

## Source-backed findings

The checkpoint contains learner, replay, planner mean and RNG, but no physical/controller state. The child constructs a new simulator and resets using the episode-index seed. The reference retains a simulator with preceding episode history. Resetting visible state is not proof of resetting hidden solver/contact state.
PD joint position reset copies current qpos into start/target and resets its step counter. get_state omits target_qpos when use_target is false, so diagnostics additionally read private target/start fields. No claim that these constitute complete simulator state.
TD-MPC2 samples policy trajectories and action noise, selects elites via top-k and samples an action via Gumbel sampling. It uses previous planner mean only when t0 is false. Small input differences could change elite selection, but this is INFERRED as a mechanism, not VERIFIED as this failure’s cause.

## Runtime experiments pending

A1: two independent fresh environments, identical recorded actions, no planner. A1_probe: separate instrumented repeats capture first physics substeps with an additional GPU fetch; probe_effect compares resulting control-step trajectory against native instrumentation.
A2: requires an original recorded warm action artifact. The original collector check discarded the warm trace; neither reference.pt, child.pt nor boundary.pt contains it. It is therefore UNRESOLVED unless another original recording is supplied. A reconstructed/newly sampled warm episode is not silently substituted.
B1: two independent model/replay/RNG restores feeding identical observations, normal t0 sequence. A separate all-t0 condition is diagnostic only. B2: reference versus child observations from identical checkpoint state. Raw CPU/all-CUDA RNG bytes and planner means are retained per call.
Exact equality and rtol=0/atol=1e-6 comparisons are both reported. Missing fields and inaccessible hidden state remain explicit limitations. No automatic cause classification or production patch follows a single comparison.

## Budget and execution

Without warm history: at most 200 environment transitions and 250 planner calls, zero updates. With one original warm episode: at most 300 environment transitions. Workers run sequentially with 180-second timeouts; external timeout 1860 seconds. No full nominal training or pilot.

## Minimal fix and acceptance

No justified new production fix identified. Preserve checkpoints, locks, upstream implementation and strict collector test. After diagnostic results establish a cause and a minimal fix is reviewed, rerun unchanged Push and Pick collector criteria. Both must pass before readiness is reconsidered. An alternative integrity-oriented protocol would require an explicit protocol decision; this tool does not substitute one.

## Exact source references

- `src/tvwm/envs/maniskill.py:74` — `def reset(self, seed):`
- `src/tvwm/envs/maniskill.py:81` — `def step(self, action):`
- `src/tvwm/agents/nominal.py:28` — `def training_seed(seed, episode):`
- `src/tvwm/agents/nominal.py:46` — `def collect(env, agent, replay, counters, seed, *, learn=True, limit=300000, on_update=None):`
- `src/tvwm/agents/resume.py:14` — `def restore_rng(state):`
- `src/tvwm/agents/resume.py:20` — `def save_learner(path, agent, replay, counters, provenance):`
- `src/tvwm/agents/resume.py:34` — `def load_learner(path, agent, replay, provenance):`
- `src/tvwm/agents/resume.py:44` — `restore_rng(state['rng'])`
- `external/ManiSkill/mani_skill/envs/sapien_env.py:861` — `def reset(self, seed: Union[None, int, list[int]] = None, options: Union[None, dict] = None) -> tuple[Any, dict]:`
- `external/ManiSkill/mani_skill/envs/sapien_env.py:964` — `self.scene._gpu_fetch_all()`
- `external/ManiSkill/mani_skill/envs/sapien_env.py:970` — `controller.reset()`
- `external/ManiSkill/mani_skill/envs/sapien_env.py:972` — `self.agent.controller.reset()`
- `external/ManiSkill/mani_skill/envs/sapien_env.py:1043` — `self.scene._gpu_fetch_all()`
- `external/ManiSkill/mani_skill/envs/sapien_env.py:1098` — `self.agent.controller.reset()`
- `external/ManiSkill/mani_skill/envs/sapien_env.py:1136` — `self.scene.step()`
- `external/ManiSkill/mani_skill/envs/sapien_env.py:1140` — `self.scene._gpu_fetch_all()`
- `external/ManiSkill/mani_skill/envs/sapien_env.py:1314` — `self.scene._gpu_fetch_all()`
- `external/ManiSkill/mani_skill/agents/controllers/pd_joint_pos.py:65` — `def reset(self):`
- `external/ManiSkill/mani_skill/agents/controllers/pd_joint_pos.py:81` — `def set_action(self, action: torch.Tensor):`
- `external/ManiSkill/mani_skill/agents/controllers/pd_joint_pos.py:108` — `def get_state(self) -> dict:`
- `external/ManiSkill/mani_skill/agents/controllers/pd_joint_pos.py:214` — `def set_action(self, action: torch.Tensor):`
- `external/tdmpc2/tdmpc2/tdmpc2.py:98` — `def act(self, obs, t0=False, eval_mode=False, task=None):`
- `external/tdmpc2/tdmpc2/tdmpc2.py:166` — `if not t0:`
- `external/tdmpc2/tdmpc2/tdmpc2.py:185` — `elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices`
- `external/tdmpc2/tdmpc2/tdmpc2.py:200` — `rand_idx = math.gumbel_softmax_sample(score.squeeze(1))`
- `external/tdmpc2/tdmpc2/tdmpc2.py:205` — `self._prev_mean.copy_(mean)`

## Server diagnostic 001 and checksum correction

VERIFIED from supplied results: A1 native control-boundary state records matched
exactly in two fresh processes. Probe control-boundary trajectories also matched;
first-substep results must be checked separately. Original input hashes unchanged.
All five planner workers stopped before their first act call: the diagnostic
model checksum assumed all state_dict entries were tensors, but torch.Size
metadata is present. This is a diagnostic implementation error, not evidence of
planner divergence. B1/B2 and causal classification remain UNRESOLVED; NO-GO persists.

The checksum now recursively encodes typed metadata and raw tensor bytes, with
regression tests for torch.Size, real TensorDict state, scalar and bfloat tensors.
Only diagnostic code changed. --planner-only repeats at most 250 act calls in
five processes, zero environment transitions and zero updates, with new outputs.

## Planner diagnostic 002: supplied server results

VERIFIED: B1 and B1_all_t0 match exactly at every compared record, with matching
model checksums and exact restored CPU/CUDA RNG states. No worker errors and
input hashes unchanged. B2, starting from the same model and restored RNG,
first differs at observation record 1; actions first exceed tolerance at index 2.
This supports observation-sequence sensitivity in the tested planner. It does not
identify the cause of the first physical difference or establish general numerical
instability. A1 previously passed; A2 remains unmeasured. Readiness remains NO-GO.

Correction to the earlier warm-history limitation: the separate warm trace was
not saved, but boundary.pt includes the completed warm episode in replay. Its
original actions may therefore be recoverable without resampling. The new
phase1_extract_warm_trace.py requires the expected three-episode checkpoint,
finds the last contiguous episode with its initial placeholder row, validates
its action data, and writes a new hash-bound warm-history artifact. It never
steps an environment, calls the planner or updates a model. Extraction is pending
server validation; no original artifact is modified.

A2-only rerun uses that exact recovered history, not newly sampled actions. It
performs one fresh target episode and one warm-plus-target sequence, at most 150
transitions and zero planner/optimizer calls. A2 failure with A1/B1 passing would
prioritize history-dependent reset/state behavior, not conclusively distinguish
controller state from hidden simulator state without examining per-field traces.
