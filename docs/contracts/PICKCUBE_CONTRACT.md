# PickCube-v1

VERIFIED source: `mani_skill/envs/tasks/tabletop/pick_cube.py`, @register_env,
and pick_cube_cfgs.py. Installed runtime: UNRESOLVED. INFERRED observation D=42:
qpos9 + qvel9 + is_grasped1 + tcp_pose7 + goal_pos3 + obj_pose7 +
tcp_to_obj_pos3 + obj_to_goal_pos3. Runtime must verify actual shapes.

Success: object-goal Euclidean distance <= goal_thresh (Panda .025m), and robot
static under native .2 velocity threshold. Grasp state affects native reward and
observation, not an extra success requirement. Dense reward uses reaching,
grasping, placement while grasped, and static bonus while placed; success reward
5; normalized_dense divides by 5. Native task definitions are preserved.

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
| Horizon | @register_env max_episode_steps=50; runtime env.spec readback |
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
