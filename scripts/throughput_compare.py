"""Bounded frozen-policy ABBA comparison. No updates, training, or checkpoint writes."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from phase1_resume_check import sha256


def trial(a):
    import torch
    from tvwm.agents.upstream import load_api,verify_sources
    from tvwm.agents.nominal import collect,empty_replay
    from tvwm.agents.resume import load_learner,rng_state
    from tvwm.envs.prefetch_env import environment_class
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    provenance=state['provenance']
    if verify_sources()!=provenance['sources']: raise ValueError('upstream mismatch')
    if sha256(ROOT/'environment/requirements-linux.lock')!=provenance['dependency_lock_sha256']:
        raise ValueError('dependency lock mismatch')
    import importlib.metadata as md
    if {k:md.version(k) for k in provenance['packages']}!=provenance['packages']: raise ValueError('package mismatch')
    Agent,Buffer=load_api()
    from common.parser import cfg_to_dataclass
    from omegaconf import OmegaConf
    cfg=cfg_to_dataclass(OmegaConf.create(provenance['config']))
    agent=Agent(cfg)
    device=state['counters'].get('replay_device') or 'cuda:0'
    replay=empty_replay(Buffer,cfg,device)
    counters=load_learner(a.checkpoint,agent,replay,provenance)
    # Identical host-only transport instrumentation in both arms; no CUDA syncs.
    import tvwm.envs.process_env as process_env
    import tvwm.envs.prefetch_env as prefetch_env
    worker_records=[]
    OriginalTransport=process_env.WorkerTransport
    class TimedTransport(OriginalTransport):
        def __init__(self,*args,**kwargs):
            self.born=time.monotonic(); self.requests=[]; self.recorded=False
            super().__init__(*args,**kwargs)
        def request(self,command,payload=None):
            started=time.monotonic()
            try: return super().request(command,payload)
            finally: self.requests.append(dict(command=command,wall_seconds=time.monotonic()-started))
        def abort(self):
            super().abort()
            if not self.recorded:
                worker_records.append(dict(pid=self.process.pid,lifetime_seconds=time.monotonic()-self.born,
                    exitcode=self.process.exitcode,alive=self.process.is_alive(),pipe_closed=self.connection.closed,
                    requests=self.requests))
                self.recorded=True
    process_env.WorkerTransport=prefetch_env.WorkerTransport=TimedTransport
    env=environment_class(a.pipeline)(provenance['task'])
    records=[]; traces=[]; initial_rng=rng_state()
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    try:
        for _ in range(2):
            t=time.monotonic()
            episode=collect(env,agent,replay,counters,1,learn=False,limit=counters['transitions']+50)
            if not episode['complete']: raise AssertionError('expected complete episode within horizon')
            traces.append(episode.pop('trace')); episode['episode_wall_seconds']=time.monotonic()-t
            records.append(episode)
    finally: env.close()
    elapsed=time.monotonic()-started
    total=sum(x['length'] for x in records)
    output=dict(pipeline=a.pipeline,workers=worker_records,episodes=records,wall_seconds=elapsed,transitions=total,
        environment_transitions_per_second=total/elapsed,optimizer_updates=0,
        learner_updates_per_second=None,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        cleanup='joined active simulator; cancelled unused CPU preparation',
        scope='frozen-policy collection, zero learner updates, excludes validation/checkpoint IO')
    torch.save(dict(traces=traces,counters=counters,initial_rng=initial_rng,final_rng=rng_state()),a.out/'trace.pt')
    (a.out/'report.json').write_text(json.dumps(output,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--pipeline',choices=['baseline','cpu-prefetch']); a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    if a.pipeline: trial(a); return
    # Never benchmark on the GPU while the current experiment is active.
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if active: raise RuntimeError('GPU processes active; leave current training untouched and defer comparison')
    original=sha256(a.checkpoint)
    summary=dict(status='NO_GO',checkpoint_sha256=original,maximum_transitions=400,optimizer_updates=0)
    try:
        runs=[]
        for i,pipeline in enumerate(('baseline','cpu-prefetch','cpu-prefetch','baseline')):
            dest=a.out/f'{i}-{pipeline}'
            with (a.out/f'{i}.log').open('w') as log:
                subprocess.run([sys.executable,__file__,'--checkpoint',str(a.checkpoint),'--out',str(dest),
                    '--pipeline',pipeline],stdout=log,stderr=subprocess.STDOUT,timeout=300,check=True)
            runs.append(json.loads((dest/'report.json').read_text()))
        import torch
        from phase1_isolation_resume import exact
        from tvwm.agents.nominal import code_hash
        reference=torch.load(a.out/'0-baseline/trace.pt',map_location='cpu',weights_only=False)
        for i,pipeline in enumerate(('baseline','cpu-prefetch','cpu-prefetch','baseline')):
            actual=torch.load(a.out/f'{i}-{pipeline}/trace.pt',map_location='cpu',weights_only=False)
            exact(actual['initial_rng'],reference['initial_rng']); exact(actual['final_rng'],reference['final_rng'])
            exact(actual['counters'],reference['counters'])
            if len(actual['traces'])!=len(reference['traces']): raise AssertionError('episode count mismatch')
            for left,right in zip(actual['traces'],reference['traces']):
                for key in right: torch.testing.assert_close(left[key],right[key],rtol=0,atol=1e-6)
        rates={}
        for pipeline in ('baseline','cpu-prefetch'):
            selected=[r for r in runs if r['pipeline']==pipeline]
            rates[pipeline]=sum(r['transitions'] for r in selected)/sum(r['wall_seconds'] for r in selected)
        gain=rates['cpu-prefetch']/rates['baseline']-1
        if sha256(a.checkpoint)!=original: raise ValueError('input checkpoint changed during comparison')
        summary.update(status='PROMISING_FROZEN_COLLECTION_ONLY' if gain>=.10 else 'REJECT_NEGLIGIBLE_GAIN',
            code_sha256=code_hash(),rates=rates,relative_gain=gain,runs=runs,
            strict_traces_and_rng='PASS',training_authorized=False,
            limitation='Two trials per pipeline; no learner updates. Full training gain and migration remain UNRESOLVED.')
    except Exception as error:
        import traceback
        summary.update(error=repr(error),traceback=traceback.format_exc())
    (a.out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    if summary['status']=='NO_GO': raise SystemExit(1)

if __name__=='__main__': main()
