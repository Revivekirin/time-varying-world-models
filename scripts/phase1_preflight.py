"""User-run compatibility gate only. No full training or 10k-transition pilot."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',required=True,choices=['PushCube-v1','PickCube-v1'])
    p.add_argument('--contract',required=True,type=Path)
    p.add_argument('--smoke',required=True,type=Path)
    p.add_argument('--evaluation',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    report=dict(task=a.task,status='RUNNING',transitions=0,updates_executed=0,
                full_training_authorized=False,tests={})
    start=time.monotonic();env=None;stop=threading.Event();samples=[];thread=None
    def timeout(*_): raise TimeoutError('preflight 600s limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    def telemetry():
        while not stop.is_set():
            try:
                text=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free,memory.total',
                    '--format=csv,noheader,nounits'],text=True,timeout=5)
                samples.append(dict(elapsed_seconds=time.monotonic()-start,readings=text.strip()))
            except Exception as e: samples.append(dict(error=repr(e)))
            stop.wait(.2)
    try:
        from tvwm.agents.upstream import verify_sources,load_api,config,episode_tensor
        pins=verify_sources()
        contract=json.loads(a.contract.read_text())
        smoke=json.loads((a.smoke/'report.json').read_text())
        evaluation=json.loads(a.evaluation.read_text())
        for record,stage in [(contract,'contract'),(smoke,'learner')]:
            if record.get('status')!='PASS' or record.get('task')!=a.task or record.get('stage')!=stage or record.get('sources')!=pins:
                raise ValueError(f'invalid {stage} evidence')
        if not smoke.get('tests',{}).get('H','').startswith('PASS'):
            raise ValueError('checkpoint smoke not passed')
        if evaluation.get('status')!='PASS' or evaluation.get('task')!=a.task or evaluation.get('updates')!=0:
            raise ValueError('independent evaluation not passed')
        episodes=evaluation.get('episodes',[])
        if [e['seed'] for e in episodes]!=[10001,10002] or not 0<evaluation['transitions']<=100:
            raise ValueError('unexpected Phase 0 evaluation budget/seeds')
        if sum(e['length'] for e in episodes)!=evaluation['transitions'] or any(not (e['terminated'] or e['truncated']) for e in episodes):
            raise ValueError('evaluation boundary/count mismatch')
        report['phase0_evaluation']=evaluation
        report['evidence_hashes']={str(path):digest(path) for path in [a.contract,a.smoke/'report.json',a.smoke/'agent.pt',a.evaluation]}
        report['tests']['phase0_evidence']='PASS user-produced reports and checkpoint present'
        report['sources']=pins
        lock_path=ROOT/'environment/requirements-linux.lock'
        report['dependency_lock_sha256']=digest(lock_path)
        check=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True,timeout=60)
        report['pip_check']=check.stdout+check.stderr
        if check.returncode: raise RuntimeError('pip check failed')
        import importlib.metadata as md
        report['packages']={name:md.version(name) for name in ['torch','numpy','gymnasium','tensordict','torchrl','sapien','mani_skill']}
        report['python']=sys.version
        report['project_revision']=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
        report['preflight_script_sha256']=digest(__file__)

        free=shutil.disk_usage(a.out).free
        report['free_storage_bytes']=free
        if free<30*2**30: raise RuntimeError('less than proposed 30GiB free storage')
        import torch
        import numpy as np
        import random
        from tvwm.envs.maniskill import ManiSkillAdapter
        from tvwm.utils.state import snapshot_state,assert_state_equal
        from tvwm.agents.resume import save_learner,load_learner
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        torch.manual_seed(1);np.random.seed(1);random.seed(1)
        torch.backends.cudnn.benchmark=False
        report['cuda_free_before_bytes'],report['cuda_total_bytes']=torch.cuda.mem_get_info()
        report['gpu_name']=torch.cuda.get_device_name(0)
        report['torch_version']=torch.__version__
        report['cuda_runtime']=torch.version.cuda
        if torch.cuda.device_count()!=1: raise RuntimeError('preflight requires one visible GPU; explicit mapping needed otherwise')
        thread=threading.Thread(target=telemetry,daemon=True);thread.start()
        torch.cuda.reset_peak_memory_stats()
        env=ManiSkillAdapter(a.task)
        obs,_=env.reset(seed=31001)
        from tvwm.envs.contracts import compare_recorded_contract
        compare_recorded_contract(env.contract, contract['contract'],
            ['obs_dim','action_dim','horizon','sim_freq','control_freq','action_mapping','reward_mode'])
        report['contract']=env.contract
        Agent,Buffer=load_api()
        cfg=config(a.task,obs.numel(),env.action_dim,env.horizon)
        cfg.batch_size=256;cfg.steps=300000;cfg.buffer_size=300000
        cfg.seed_steps=1000
        (a.out/'resolved_config.json').write_text(json.dumps(vars(cfg),indent=2,default=str))
        agent=Agent(cfg);agent.load(a.smoke/'agent.pt')
        before=snapshot_state(agent.model.state_dict())
        action=agent.act(obs,t0=True,eval_mode=True)
        if not torch.isfinite(action).all() or action.abs().max()>1: raise ValueError('invalid frozen action')
        assert_state_equal(agent.model.state_dict(),before)
        report['tests']['checkpoint_frozen_forward']='PASS'
        with torch.no_grad():
            z=agent.model.encode(obs[None].cuda(),None)
            reward=torch.ones(1,1,device='cuda')
            terminal=torch.ones_like(reward)
            target=agent._td_target(z,reward,terminal,None)
            torch.testing.assert_close(target,reward,rtol=0,atol=0)
        report['tests']['terminal_bootstrap_mask']='PASS upstream target equals reward for terminated=1'

        del before
        replay=Buffer(cfg)
        for episode in range(2):
            obs,_=env.reset(seed=31001+episode)
            observations=[obs];actions=[];rewards=[];terms=[];truncs=[]
            for _ in range(50):
                act=torch.rand(8)*2-1
                obs,r,term,trunc,_=env.step(act);report['transitions']+=1
                observations.append(obs);actions.append(act);rewards.append(r);terms.append(term);truncs.append(trunc)
                if term or trunc: break
            if len(actions)<3: raise RuntimeError('episode shorter than replay horizon; stop without resampling')
            replay.add(episode_tensor(observations,actions,rewards,terms,truncs))
        report['replay_storage_device']=str(replay._storage_device)
        report['replay_capacity_rows']=replay.capacity
        report['losses']=[];report['update_seconds']=[]
        def update():
            torch.cuda.synchronize();t=time.monotonic()
            metrics=agent.update(replay)
            torch.cuda.synchronize();report['updates_executed']+=1
            report['update_seconds'].append(time.monotonic()-t)
            row={k:float(v) for k,v in metrics.items()}
            if not all(math.isfinite(v) for v in row.values()): raise FloatingPointError('nonfinite batch256 metric')
            report['losses'].append(row)
            return row
        update()
        provenance=dict(sources=pins,config=vars(cfg),task=a.task,dependency_lock_sha256=report['dependency_lock_sha256'],packages=report['packages'])
        counters=dict(transitions=report['transitions'],updates=1,episode_index=2)
        checkpoint=a.out/'learner-envelope.pt'
        save_learner(checkpoint,agent,replay,counters,provenance)
        reference=update();state=snapshot_state(agent.model.state_dict())
        restored=load_learner(checkpoint,agent,replay,provenance)
        if restored!=counters: raise AssertionError('counter restore mismatch')
        resumed=update()
        for key in reference:
            if not math.isclose(reference[key],resumed[key],rel_tol=1e-5,abs_tol=1e-6):
                raise AssertionError(f'resumed next update metric mismatch: {key}')
        # Strict parameters detect replay/optimizer/RNG restore divergence.
        assert_state_equal(agent.model.state_dict(),state)
        report['tests']['batch256_forward_backward_optimizer']='PASS 3 physical updates, including replayed next update'
        report['tests']['learner_envelope']='PASS same-process replay/optimizers/scale/RNG/counters; collector and fresh-process resume NOT tested'
        report['checkpoint_sha256']=digest(checkpoint)
        report['cuda_free_after_bytes'],_=torch.cuda.mem_get_info()
        report['torch_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        report['torch_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        if report['torch_peak_reserved_bytes']>20*2**30 or report['cuda_free_after_bytes']<4*2**30:
            raise RuntimeError('proposed 20GiB budget / 4GiB free headroom violated')
        report['status']='PASS_COMPATIBILITY_ONLY'
    except Exception as e:
        report.update(status='FAIL',error=repr(e),traceback=traceback.format_exc())
    finally:
        stop.set()
        if thread is not None:thread.join(timeout=6)
        report['gpu_telemetry']=samples
        valid=[s for s in samples if 'readings' in s]
        if not valid and report['status']=='PASS_COMPATIBILITY_ONLY':
            report.update(status='FAIL',error='device-wide GPU telemetry missing')
        if valid:
            used=[float(line.split(',')[1]) for s in valid for line in s['readings'].splitlines()]
            report['sampled_device_peak_used_mib']=max(used)
            if max(used)>20*1024 and report['status']=='PASS_COMPATIBILITY_ONLY':
                report.update(status='FAIL',error='device-wide sampled memory exceeds 20GiB')
        report['wall_seconds']=time.monotonic()-start
        (a.out/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
        if env is not None:env.close()
        signal.alarm(0)
    print(json.dumps(report,indent=2,default=str))
    return 0 if report['status']=='PASS_COMPATIBILITY_ONLY' else 1

if __name__=='__main__':sys.exit(main())
