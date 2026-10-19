# Upstream source audit

VERIFIED (local source inspection, not runtime). Exact checked-out commits:
- tdmpc2: `e9f59321933cbc8e11a002b842adc7d4ffae8ff1` — https://github.com/nicklashansen/tdmpc2.git
- tdmpc: `f4d85eca7419039b71bab2234ffc8aca378dd313` — https://github.com/nicklashansen/tdmpc.git
- ManiSkill: `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3` — https://github.com/haosulab/ManiSkill.git

No patches. Checkout tree paths below are relative to each upstream root.
Source provenance lives in `environment/sources.lock.json`; ignored external
checkouts are recreated by the fetch script. All three repositories were fetched
from official URLs; ManiSkill's old organization redirects to mani-skill.

## TD-MPC2

| Component | Inspected source | Findings |
|---|---|---|
| Encoder | tdmpc2/common/layers.py: enc, mlp, SimNorm | State MLP, LayerNorm/Mish and final simplicial normalization; 5M preset uses enc_dim 256, latent_dim 512, two encoder layers. |
| Dynamics | tdmpc2/common/world_model.py: next | Concatenated latent/action, two hidden MLP layers, SimNorm output. Single-task task_dim=0. |
| Reward | world_model.py: reward; common/math.py | Distributional reward logits, decoded with upstream two-hot inverse; no adapter reward scaling. |
| Q/targets | world_model.py: Q, init, soft_update_target_Q | Five Qs by default, vectorized parameters; randomly selected pair for min/avg; detached shared Q and cloned target Q parameters; Polyak tau=.01. |
| Policy | world_model.py: pi; tdmpc2.py: update_pi | Squashed Gaussian, bounded log std, entropy term and running Q scale; rho-weighted objective. |
| Planner | tdmpc2/tdmpc2.py: _plan, _estimate_value | Horizon=3, 6 iterations, 512 samples, 64 elites, 24 policy trajectories; exponential elite weighting, warm-start mean; Gumbel elite choice, added exploration noise outside eval. Actions clamped. |
| Replay | tdmpc2/common/buffer.py | TorchRL SliceSampler with episode trajectory key, strict horizon+1 slices, row-0 action/reward/termination discarded. CPU or CUDA storage heuristic; sampled learner tensors on cuda:0. |
| Objectives | tdmpc2/tdmpc2.py: _td_target, _update | Stop-gradient next encoding, latent consistency MSE; reward/Q soft CE, rho-weighting; BCE termination head when episodic. TD reward + discount*(1-terminated)*target-Q. Model/actor optimizers and clipping preserved. |
| Runner | tdmpc2/train.py, trainer/online_trainer.py | CUDA required. Completed episodes inserted on next reset. Seed pretraining burst and inclusive steps loop are unsuitable for a strictly bounded smoke budget. External runner uses explicit counts instead. |
| Evaluation | tdmpc2/evaluate.py; trainer/online_trainer.py: eval | Standalone evaluate calls act without eval_mode=True; trainer eval passes True. Our frozen evaluation explicitly passes True, still stochastic due to planner sampling. |
| Checkpoint | tdmpc2/tdmpc2.py: save/load | Saves model.state_dict including Q targets, not optimizers, replay, running scale, planner cache or RNG; evaluation checkpoint, not exact training resume. Legacy API conversion exists. |
| Episodes | same agent, world_model, buffer, trainer | episodic defaults false. True adds termination head/BCE and predicted termination masking in planning. Learner bootstrap uses observed terminated; truncation must remain separate. Runner rejects true terminations with episodic false. |
| ManiSkill2 | tdmpc2/envs/maniskill.py | imports mani_skill2; PickCube-v0, pd_ee_delta_pos; repeats actions twice, sums rewards, adapts old four-return API and timeout 100. It is not used for ManiSkill3. |

Upstream discount uses clamp(1-5/episode_length, .95, .995): for native 50-step
horizons, .95. We retain this rather than substitute an arbitrary discount.
`layers.enc` uses a mutable default dictionary: avoid multiple task agents in
one process; all provided task commands run separate processes. No patch needed
for this isolated, state-only usage. Generic top-level `common` imports are also
process-isolated and checked before adding upstream sys.path.

## Original TD-MPC comparison (not installed/trained)

Inspected `src/algorithm/tdmpc.py`, `helper.py`, `src/train.py`, environment.yaml.
TOLD exposes h/next/pi/Q, with next returning latent and scalar reward together;
TD-MPC2 exposes encode/next/reward separately. Original uses two scalar Qs,
a deterministic tanh policy with sampling noise, a full copied model target,
MSE losses and prioritized replay/importance weighting. The helper buffer uses
fixed episode storage and samples future observations/actions from indices.
Original update takes `(replay_buffer, step)`; TD-MPC2 takes `buffer` and uses
sequence TensorDict sampling. Original source TD target has no terminated mask.
Original checkpoint contains model and model_target, also not optimizer/replay.
Its historical Python 3.8/PyTorch 1.9/CUDA11.1/Gym0.21 stack is kept separate.
Do not transplant its replay, target model, or loss into TD-MPC2.

UNRESOLVED: all model/planner/optimizer behavior in the target Linux environment.
Source inspection does not establish numerical stability or task competence.
