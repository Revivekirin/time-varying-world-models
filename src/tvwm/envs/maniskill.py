"""Single-environment bridge. Native batched GPU API, explicit terminal boundaries."""
from dataclasses import asdict
import json
from pathlib import Path
import torch
from tvwm.envs.contracts import TASKS


def observation(value):
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != 1:
        raise ValueError('expected native state tensor [1, D]')
    if not torch.isfinite(value).all():
        raise ValueError('nonfinite observation')
    return value[0].to(dtype=torch.float32).clone()


class ManiSkillAdapter:
    def __init__(self, task):
        if task not in TASKS:
            raise ValueError(task)
        import gymnasium as gym
        import mani_skill.envs  # registration
        root = Path(__file__).resolve().parents[3]
        if not Path(mani_skill.__file__).resolve().is_relative_to((root/'external/ManiSkill').resolve()):
            raise RuntimeError('ManiSkill import must use the pinned editable checkout')
        settings = json.loads((root/f'configs/env/{task}.json').read_text())
        if settings.pop('task') != task or settings.pop('action_repeat') != 1:
            raise ValueError('unsupported task/repeat configuration')
        if settings != dict(num_envs=1, robot_uids='panda', obs_mode='state', control_mode='pd_joint_delta_pos', reward_mode='normalized_dense'):
            raise ValueError('Phase 0 manifest requires the audited environment settings')
        self.env = gym.make(task, **settings, sim_backend='gpu', render_mode=None)
        self.base = self.env.unwrapped
        self.task = task
        self.done = True
        self.horizon = self.env.spec.max_episode_steps
        if self.horizon != 50 or self.base.sim_freq != 100 or self.base.control_freq != 20:
            self.env.close()
            raise ValueError("installed horizon/frequencies differ from bounded manifest")
        self.space = self.base.single_action_space
        self.action_dim = self.space.shape[0]
        controller = self.base.agent.controller
        arm = controller.controllers['arm'].config
        gripper = controller.controllers['gripper'].config
        # Fail closed if this installed revision changes command meaning.
        if not (arm.use_delta and not arm.use_target and not arm.interpolate and arm.normalize_action):
            raise ValueError('unsupported arm semantics')
        if arm.lower != -0.1 or arm.upper != 0.1:
            raise ValueError('unexpected physical arm delta bounds')
        if gripper.use_delta or not gripper.normalize_action or gripper.lower != -0.01 or gripper.upper != 0.04:
            raise ValueError('expected normalized absolute mimic gripper with audited physical bounds')
        if dict(controller.action_mapping) != {'arm': (0, 7), 'gripper': (7, 8)}:
            # Some revisions use lists instead of tuples.
            if {k: list(v) for k, v in controller.action_mapping.items()} != {'arm': [0, 7], 'gripper': [7, 8]}:
                raise ValueError('unexpected action component mapping')
        if self.action_dim != 8 or not (self.space.low == -1).all() or not (self.space.high == 1).all():
            raise ValueError('expected normalized Panda delta action space')
        self.contract = dict(task=task, horizon=self.horizon, action_dim=self.action_dim,
            action_low=self.space.low.tolist(), action_high=self.space.high.tolist(),
            supported_control_modes=self.base.agent.supported_control_modes,
            action_mapping=controller.action_mapping, arm_config=asdict(arm),
            gripper_config=asdict(gripper), sim_freq=self.base.sim_freq,
            control_freq=self.base.control_freq, action_repeat=1,
            device=str(self.base.device), gpu_sim_enabled=self.base.gpu_sim_enabled,
            reward_mode='normalized_dense')

    def reset(self, seed):
        obs, info = self.env.reset(seed=seed)
        self.done = False
        obs = observation(obs)
        self.contract.update(obs_dim=obs.numel(), obs_dtype=str(obs.dtype))
        return obs, info

    def step(self, action):
        if self.done:
            raise RuntimeError('explicit reset required after terminated or truncated')
        action = torch.as_tensor(action, device=self.base.device, dtype=torch.float32)
        if action.shape != (self.action_dim,) or not torch.isfinite(action).all():
            raise ValueError('invalid action')
        # Both interfaces already use [-1, 1]; no second physical-unit scaling.
        if torch.any(action.abs() > 1):
            raise ValueError('out-of-bounds requested action')
        obs, reward, terminated, truncated, info = self.env.step(action.unsqueeze(0))
        if not torch.isfinite(reward).all():
            raise ValueError('nonfinite reward')
        self.done = bool(terminated.item() or truncated.item())
        return observation(obs), reward.reshape(()).clone(), bool(terminated.item()), bool(truncated.item()), info

    def close(self):
        self.env.close()
