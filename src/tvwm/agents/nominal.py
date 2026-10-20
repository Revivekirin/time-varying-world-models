"""Shared nominal episode collector; checkpoints are taken only between episodes."""
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from tvwm.agents.cadence import updates_at, random_action_at
from tvwm.agents.upstream import episode_tensor

ROOT = Path(__file__).resolve().parents[3]


def code_hash():
    h = hashlib.sha256()
    paths = sorted((ROOT/'src/tvwm').rglob('*.py')) + sorted((ROOT/'scripts').glob('phase1*.py'))
    paths += sorted((ROOT/'configs').rglob('*.json'))
    for p in paths:
        h.update(str(p.relative_to(ROOT)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def training_seed(seed, episode):
    return 1000000 + seed * 100000 + episode


def goal_distance(env):
    if env.task == 'PushCube-v1':
        return float(torch.linalg.vector_norm(env.base.obj.pose.p[..., :2]-env.base.goal_region.pose.p[..., :2]))
    return float(torch.linalg.vector_norm(env.base.cube.pose.p-env.base.goal_site.pose.p))


def append_episode(replay, episode):
    """Match initialized storage explicitly, including after TorchRL state restoration."""
    device = getattr(replay, '_storage_device', None)
    if device is not None:
        episode = episode.to(device)
    return replay.add(episode)


def collect(env, agent, replay, counters, seed, *, learn=True, limit=300000, on_update=None):
    """Update from completed replay, then insert the newly completed episode (upstream order)."""
    obs, _ = env.reset(seed=training_seed(seed, counters['episode_index']))
    observations=[obs.clone()]; actions=[]; rewards=[]; terms=[]; truncs=[]
    start_distance=goal_distance(env); planner_seconds=0.; update_seconds=0.; success=False
    while counters['transitions'] < limit:
        step=counters['transitions']; t=time.monotonic()
        action = (torch.rand(env.action_dim)*2-1 if learn and random_action_at(step)
                  else agent.act(obs, t0=len(actions)==0, eval_mode=False))
        torch.cuda.synchronize(); planner_seconds += time.monotonic()-t
        obs, reward, term, trunc, info=env.step(action)
        observations.append(obs.clone()); actions.append(action.clone()); rewards.append(float(reward))
        terms.append(term); truncs.append(trunc); success |= bool(info['success'].item())
        counters['transitions'] += 1
        if learn:
            for _ in range(updates_at(step)):
                t=time.monotonic(); metrics={k:float(v) for k,v in agent.update(replay).items()}
                torch.cuda.synchronize(); elapsed=time.monotonic()-t; update_seconds+=elapsed
                if not all(np.isfinite(v) for v in metrics.values()): raise ValueError('nonfinite learner metric')
                counters['updates'] += 1
                if on_update: on_update(dict(transitions=counters['transitions'],updates=counters['updates'],seconds=elapsed,**metrics))
        if term or trunc: break
    complete=bool(terms and (terms[-1] or truncs[-1]))
    if complete:
        append_episode(replay, episode_tensor(observations,actions,rewards,terms,truncs))
        counters['episode_index'] += 1
    return dict(environment_recreation_seconds=getattr(env,'last_recreation_seconds',0.),
                environment_reset_seconds=getattr(env,'last_reset_seconds',0.),length=len(actions),return_=sum(rewards),success=success,complete=complete,
                goal_progress=start_distance-goal_distance(env),final_goal_distance=goal_distance(env),
                planner_seconds=planner_seconds,update_seconds=update_seconds,
                trace=dict(obs=torch.stack(observations).cpu(),actions=torch.stack(actions).cpu(),
                           rewards=torch.tensor(rewards),terminated=torch.tensor(terms),truncated=torch.tensor(truncs)))


def empty_replay(Buffer, cfg, device):
    from torchrl.data.replay_buffers import LazyTensorStorage
    replay=Buffer(cfg); replay._storage_device=torch.device(device)
    replay._buffer=replay._reserve_buffer(LazyTensorStorage(replay.capacity,device=replay._storage_device))
    return replay


def atomic_json(path, value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); tmp.replace(path)
