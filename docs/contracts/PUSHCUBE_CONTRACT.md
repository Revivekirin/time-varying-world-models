# PushCube-v1

VERIFIED source: `mani_skill/envs/tasks/tabletop/push_cube.py`, @register_env.
Installed runtime: UNRESOLVED. INFERRED observation dimension 35:
qpos9 + qvel9 + tcp_pose7 + goal_pos3 + obj_pose7, no controller state in selected mode.
Runtime must report actual leaf shapes and D rather than trust this arithmetic.

Success: cube xy distance to goal < .1m and cube z < .025m. Dense reward combines
reaching the push pose, placement gated on reaching within .01m, and a z-placement
term. Success overrides reward to 4; normalized_dense divides by 4. An outdated
comment mentions 3, but executable assignment is 4. No native definition changed.

## Shared contract (source-verified; runtime UNRESOLVED)

| Item | Contract / source evidence |
|---|---|
| Revision | environment/sources.lock.json; runtime must import this editable checkout, clean tree |
| Robot/control | panda / pd_joint_delta_pos; verify installed configs and action_mapping |
| Action | normalized float32 [8], native step [1,8]; seven arm components then one mimic gripper |
| Arm | normalized [-1,1] maps to [-.1,.1] radians; target = measured current qpos + delta; use_target=false; interpolate=false |
| Gripper | absolute mimic position [-.01,.04] m via normalized [-1,1], not delta; never indiscriminately scale |
| Saturation | native BaseController clips/scales; adapter rejects out-of-range/nonfinite input to expose mismatch; valid planner outputs unchanged |
| Observation | native flat state [1,D] to cloned float32 [D]; native flatten order retained, no info concatenation |
| Frequency | SimConfig defaults 100Hz simulation, 20Hz control; 5 physics substeps/control transition; runtime records actual values |
| Repeat | exactly one native env.step per requested action; not old ManiSkill2 repeat=2 |
| Reward | native normalized_dense selected explicitly; unchanged native formula/scale |
| Boundary | native success/fail terminated; TimeLimitWrapper truncated at 50; reset on either; bootstrap masks terminated only |
| Horizon | @register_env max_episode_steps=50; runtime native find_max_episode_steps_value wrapper readback |
| State restore | get_state_dict/set_state_dict available; setter does not itself restore controller, elapsed counter or RNG; explicitly call agent.set_controller_state when needed |
| Memory | delta controller resets start/target qpos; no persistent target integration for chosen mode; other target-delta modes are different |
| GPU | native outputs batched on device; initial integration intentionally num_envs=1 because upstream planner has one warm-start cache; multi-env reset/replay not validated |
| Reproducibility | two same-seed resets compared with atol=1e-6; not yet executed; no bitwise GPU determinism promise |
| Devices | environment CUDA float tensors; adapter action moved to env device, observation float32; upstream act returns CPU; replay converts to cuda:0 |

Supported Panda modes in inspected source: pd_joint_delta_pos, pd_joint_pos,
pd_ee_delta_pos, pd_ee_delta_pose, pd_ee_pose, pd_joint_target_delta_pos,
pd_ee_target_delta_pos, pd_ee_target_delta_pose, pd_joint_vel,
pd_joint_pos_vel, pd_joint_delta_pos_vel. Runtime report enumerates installed modes.

Sources: mani_skill/envs/sapien_env.py; utils/registration.py; utils/structs/types.py;
agents/robots/panda/panda.py; agents/controllers/base_controller.py and
pd_joint_pos.py; agents/base_agent.py. No dynamics parameters or schedule labels
are added to agent observations. Later interventions require an observation-field
leakage audit, especially any controller state; task goal/object pose is native
state information, not hidden dynamics context.

`contract` smoke checks same-seed initial observations, identity state roundtrip,
zero/small alternating joint actions, finite outputs, boundaries, dimensions and
controller semantics. Identity roundtrip does not establish reproducible branched
future trajectories. Full state/RNG/time restoration and N>1 GPU behavior remain
UNRESOLVED, and are explicit readiness gaps if required by later experiments.

## Runtime evidence update

VERIFIED by user-run server2 report `results/push-contract-002/report.json`: bounded
A/B gate passed with100 transitions,4 explicit resets,0 learner updates.
Observed flattened dimension35 float32; native action8, bounds[-1,1],
100Hz/20Hz, horizon50 via native helper (env.spec horizon null), GPU enabled.
Same-seed reset max difference0, identity state roundtrip passed. Both50-step
episodes truncated without success/termination. This establishes the bounded
single-env contract gate, not success termination coverage, exact future-state
restoration, multi-env behavior or nominal competence.
