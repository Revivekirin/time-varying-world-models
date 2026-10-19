# Future dynamics intervention feasibility

Source-only audit; **no intervention implementation or shifted transitions**.
Primary candidate: hidden arm delta-command effectiveness. At each control step,
future a_exec_arm = g[t] * a_requested_arm. Keep gripper command unchanged. Chosen
Panda pd_joint_delta_pos has verified arm delta semantics; absolute pd_joint_pos
cannot be substituted. Current code sends requested actions unchanged.

| Mechanism | Source evidence | Mid-episode conclusion |
|---|---|---|
| A: command scaling | CombinedController action_mapping; PDJointPos set_action adds scaled delta to measured qpos | INFERRED feasible externally each step; installed/runtime effect untested |
| B: drive stiffness/damping | PDJointPosController.set_drive_property; structs/articulation_joint.py set_drive_properties forwards to wrapped SAPIEN joints | VERIFIED API exists; GPU post-initialization application/readback is UNRESOLVED; no assumption that config assignment changes physics |
| C: torque/force effectiveness | articulation set_qf/get_qf, set_joint_drive_targets; joint force_limit and drive modes | VERIFIED API exists, but force limit, applied external qf, and motor-output scaling are different; multiplicative actual drive-torque hook UNRESOLVED |

A modifies command amplitude, B modifies closed-loop PD response, C modifies
actuation. A does not reproduce real motor degradation. Force-limit modification
is saturation, not generally multiplicative torque effectiveness. GPU buffers and
cached joint properties may require synchronization; installed SAPIEN execution
must establish when changes reach physics and whether per-env values are possible.

Native readbacks available in source: qpos/qvel, joint target positions/velocities,
qf and incoming joint forces. qf must not be assumed to equal actual motor torque;
incoming joint forces also include constraints/contact. Controller target memory,
clipping and joint limits can hide changes in requested command gain. Target-delta
mode integrates memory; selected measured-qpos delta mode does not. State setter
alone does not restore all controller/RNG/time state.

Future schedule contract (specification only): S0 nominal g=1; S1 fixed OOD scalar;
S2 piecewise abrupt change at an explicit control-step index; S3 continuous gradual
drift over a specified interval; S4 continuous departure and return. Indices are
per episode and causal, applied before the native controller call. Log requested
and executed arm commands, saturation masks and privileged schedule separately
from agent observations. No final gain, drift duration or change-time severity is
chosen. Preserve baseline/shift matched state-action trials and measure actual
transition displacement before declaring a valid mechanism. Establish controller,
RNG and simulation-state restoration before paired counterfactual experiments.

Future validity tests need free-motion and contact regimes, native readbacks,
saturation analysis, zero-change negative control, repetitions and severity curves.
They belong to Phase 2 after nominal readiness, not this Phase 0 runner.
