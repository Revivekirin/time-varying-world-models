"""Bounded user-run lifecycle-only crash probe; zero actions/learner/planner calls.
Separate script intentionally leaves checkpoint-bound production files unchanged.
"""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cycles',type=int,default=1000)
    a=p.parse_args()
    if not 1<=a.cycles<=1000:p.error('cycles must be 1..1000')
    a.out.mkdir(parents=True,exist_ok=False)
    stream=(a.out/'stages.jsonl').open('w',buffering=1)
    fault=(a.out/'faulthandler.log').open('w')
    faulthandler.enable(file=fault,all_threads=True)
    started=time.monotonic();cycle=0
    def log(stage,**extra):
        stream.write(json.dumps(dict(cycle=cycle,stage=stage,seconds=time.monotonic()-started,**extra))+'\n')
        stream.flush();os.fsync(stream.fileno())
    def interrupted(*_):raise TimeoutError('lifecycle diagnostic deadline or interruption')
    signal.signal(signal.SIGALRM,interrupted);signal.alarm(2370)
    signal.signal(signal.SIGTERM,interrupted)
    log('imports.begin')
    import torch
    from tvwm.envs import episode_lifecycle as lifecycle
    from tvwm.agents.upstream import verify_sources
    from tvwm.agents.nominal import training_seed,code_hash
    pins=verify_sources()
    manifest=dict(task='PushCube-v1',max_cycles=a.cycles,simulation_transitions=0,
                  learner_updates=0,planner_calls=0,max_wall_seconds=2370,
                  sources=pins,production_code_sha256=code_hash(),
                  limitation='No physics steps or learner allocations. A pass does not establish stability of full training.')
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def instrument(name,fn):
        def call(*args,**kwargs):
            log(name+'.begin')
            result=fn(*args,**kwargs)
            log(name+'.end')
            return result
        return call
    from tvwm.envs.maniskill import ManiSkillAdapter
    original_factory=lifecycle.adapter_factory
    original_sync=lifecycle.synchronize
    original_gc=lifecycle.gc.collect
    original_reset=ManiSkillAdapter.reset
    original_close=ManiSkillAdapter.close
    # Class-level wrappers avoid adding bound-method reference cycles per adapter.
    ManiSkillAdapter.reset=instrument('adapter.reset',original_reset)
    ManiSkillAdapter.close=instrument('adapter.close',original_close)
    lifecycle.adapter_factory=instrument('adapter.construct',original_factory)
    lifecycle.synchronize=instrument('cuda.synchronize',original_sync)
    lifecycle.gc.collect=instrument('gc.collect',original_gc)
    env=lifecycle.EpisodeEnvironment('PushCube-v1')
    def resources():
        status={}
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(('VmRSS:','VmSize:','Threads:')):
                key,value=line.split(':',1);status[key]=value.strip()
        free,total=torch.cuda.mem_get_info()
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
        if len(rows)!=1:raise RuntimeError('requires audited single physical GPU')
        used,available=[int(x.strip()) for x in rows[0].split(',')]
        log('resources',proc=status,fd_count=len(list(Path('/proc/self/fd').iterdir())),
            cuda_free_bytes=free,cuda_total_bytes=total,device_used_mib=used,
            torch_allocated_bytes=torch.cuda.memory_allocated(),torch_reserved_bytes=torch.cuda.memory_reserved())
        if used>20*1024 or min(free,available*1024**2)<4*1024**3:
            raise RuntimeError('existing GPU resource budget exceeded')
    result=dict(status='FAIL',completed_cycles=0,simulation_transitions=0,learner_updates=0,planner_calls=0)
    try:
        resources()
        for cycle in range(a.cycles):
            if time.monotonic()-started>=2300:raise TimeoutError('shutdown reserve reached')
            seed=training_seed(1,cycle)
            log('episode_recreation.begin',seed=seed)
            env.reset(seed)
            log('episode_recreation.end',seed=seed)
            result['completed_cycles']=cycle+1
            if (cycle+1)%10==0:resources()
            if (cycle+1)%50==0:print(f'{cycle+1}/{a.cycles} lifecycle cycles completed; zero simulation steps',flush=True)
        result['status']='PASS_LIFECYCLE_ONLY'
    except Exception as error:
        result['error']=repr(error)
        log('exception',error=repr(error))
    finally:
        log('final_close.begin')
        env.close()
        log('final_close.end')
        result['wall_seconds']=time.monotonic()-started
        (a.out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        lifecycle.adapter_factory=original_factory;lifecycle.synchronize=original_sync
        lifecycle.gc.collect=original_gc
        ManiSkillAdapter.reset=original_reset;ManiSkillAdapter.close=original_close
        signal.alarm(0);stream.close();faulthandler.disable();fault.close()
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='PASS_LIFECYCLE_ONLY' else 1

if __name__=='__main__':sys.exit(main())
