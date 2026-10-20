"""Recover original warm actions from boundary replay; no simulator, planner, or update."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from phase1_root_cause import digest,write


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    import torch
    from tvwm.agents.upstream import load_api
    from tvwm.agents.nominal import empty_replay,training_seed
    checkpoint_path=a.source/'boundary.pt'; original_hash=digest(checkpoint_path)
    state=torch.load(checkpoint_path,weights_only=False)
    # The existing check loads two preflight episodes then inserts one warm episode.
    counters=state['counters']; num_eps=state['replay_num_eps']
    if counters['episode_index']!=3 or num_eps!=3 or not 0<counters['transitions']<=150:
        raise ValueError('not the expected original three-episode boundary; stop rather than guess history')
    _,Buffer=load_api()
    from common.parser import cfg_to_dataclass
    from omegaconf import OmegaConf
    cfg=cfg_to_dataclass(OmegaConf.create(state['provenance']['config']))
    replay=empty_replay(Buffer,cfg,'cuda:0')
    replay._buffer.load_state_dict(state['replay'])
    storage=replay._buffer._storage
    rows=storage[:len(replay._buffer)]
    ids=rows['episode'].reshape(-1)
    positions=(ids==num_eps-1).nonzero().reshape(-1)
    if not 2<=len(positions)<=51 or not torch.equal(positions[1:],positions[:-1]+1):
        raise ValueError('warm episode missing, overwritten, or noncontiguous')
    episode=rows[positions].cpu()
    if not torch.isnan(episode['action'][0]).all() or not torch.isnan(episode['reward'][0]).all():
        raise ValueError('missing initial placeholder row')
    actions=episode['action'][1:].clone()
    if actions.shape[1:]!=(8,) or not torch.isfinite(actions).all() or (actions.abs()>1).any():
        raise ValueError('invalid recorded actions')
    trace=dict(obs=episode['obs'].clone(),actions=actions,rewards=episode['reward'][1:].clone(),
               terminated=episode['terminated'][1:].clone())
    artifact=a.out/'original-warm.pt'
    torch.save(dict(trace=trace,provenance=dict(boundary_sha256=original_hash,episode_id=num_eps-1,
                                              storage_indices=positions.cpu().tolist())),artifact)
    manifest=dict(original_recording=True,extraction_source='saved completed replay episode, no resampling',
                  boundary_sha256=original_hash,episodes=[dict(path=artifact.name,sha256=digest(artifact),
                  seed=training_seed(1,counters['episode_index']-1))])
    write(a.out/'warm-history.json',manifest)
    if digest(checkpoint_path)!=original_hash: raise AssertionError('input checkpoint changed')
    write(a.out/'report.json',dict(status='EXTRACTED_ORIGINAL_REPLAY_ACTIONS',simulation_transitions=0,
          planner_calls=0,optimizer_updates=0,episode_id=num_eps-1,recorded_actions=len(actions),
          boundary_sha256=original_hash,limitation='Replay does not store truncation; original collector admitted only completed episodes. A2 checks actual replayed boundary.'))
    print((a.out/'report.json').read_text())

if __name__=='__main__':main()
