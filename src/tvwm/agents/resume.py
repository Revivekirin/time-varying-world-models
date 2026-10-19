"""Phase 1 learner checkpoint envelope. Collector state is a separate launch gate."""
import os
from pathlib import Path
import random
import numpy as np
import torch


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def save_learner(path, agent, replay, counters, provenance):
    """Atomic publication; preserves old destination if writing the new file fails."""
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    state = dict(format_version=1, model=agent.model.state_dict(),
                 optim=agent.optim.state_dict(), pi_optim=agent.pi_optim.state_dict(),
                 scale=agent.scale.state_dict(), planner_mean=agent._prev_mean,
                 replay=replay._buffer.state_dict(), replay_num_eps=replay._num_eps,
                 counters=counters, provenance=provenance, rng=rng_state())
    with tmp.open('wb') as f:
        torch.save(state, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def load_learner(path, agent, replay, provenance):
    """Trusted local checkpoints only; replay storage must already be initialized."""
    state = torch.load(path, weights_only=False)
    if state['format_version'] != 1 or state['provenance'] != provenance:
        raise ValueError('checkpoint format/provenance mismatch')
    agent.model.load_state_dict(state['model'])
    agent.optim.load_state_dict(state['optim']); agent.pi_optim.load_state_dict(state['pi_optim'])
    agent.scale.load_state_dict(state['scale'])
    agent._prev_mean.copy_(state['planner_mean'])
    replay._buffer.load_state_dict(state['replay']); replay._num_eps=state['replay_num_eps']
    restore_rng(state['rng'])
    return state['counters']
