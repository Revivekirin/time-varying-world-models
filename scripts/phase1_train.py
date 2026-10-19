"""Gated, exact-budget nominal TD-MPC2 runner. Remote execution is user initiated."""
import argparse
import importlib.metadata as md
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from phase1_resume_check import sha256


def read(path): return json.loads(Path(path).read_text())


def resources(out, plan):
    import torch
    free,total=torch.cuda.mem_get_info()
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
    # One visible physical GPU only: avoid checking an unrelated GPU by mistake.
    lines=rows.strip().splitlines()
    if len(lines)!=1: raise RuntimeError('single physical GPU required for audited resource mapping')
    index,used,available=[int(x.strip()) for x in lines[0].split(',')]
    disk=shutil.disk_usage(out).free
    if used*1024**2>plan['gpu_budget_gib']*1024**3 or min(free,available*1024**2)<4*1024**3:
        raise RuntimeError('GPU budget/free-memory guard failed')
    if disk<plan['minimum_free_storage_gib']*1024**3: raise RuntimeError('free-storage guard failed')
    return dict(gpu_index=index,device_used_mib=used,device_free_mib=available,cuda_free_bytes=free,
                cuda_total_bytes=total,storage_free_bytes=disk,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())


def evaluate(a):
    import torch
    from tvwm.agents.upstream import load_api
    from tvwm.agents.nominal import seed_all,goal_distance,atomic_json
    from tvwm.envs.maniskill import ManiSkillAdapter
    Agent,_=load_api()
    from common.parser import cfg_to_dataclass
    from omegaconf import OmegaConf
    cfg=cfg_to_dataclass(OmegaConf.create(read(a.out/'resolved_config.json')))
    env=ManiSkillAdapter(cfg.task); agent=Agent(cfg); agent.load(a.out/('best-model.pt' if a.heldout else 'evaluation-model.pt'))
    agent.model.eval(); agent.model.requires_grad_(False)
    episodes=[]; steps=0; planner=0.
    plan=read(a.out/'nominal_config.json')
    seeds=(range(plan['heldout_seed_start'],plan['heldout_seed_start']+plan['heldout_episodes'])
           if a.heldout else plan['validation_seeds'])
    for seed in seeds:
        seed_all(seed); obs,_=env.reset(seed=seed); initial=goal_distance(env); ret=0.; success=False
        for t in range(env.horizon):
            started=time.monotonic(); action=agent.act(obs,t0=t==0,eval_mode=True)
            torch.cuda.synchronize(); planner+=time.monotonic()-started
            obs,r,term,trunc,info=env.step(action); ret+=float(r); steps+=1; success|=bool(info['success'].item())
            if term or trunc: break
        else: raise AssertionError('evaluation missing episode boundary')
        episodes.append(dict(seed=seed,length=t+1,return_=ret,success=success,
                             terminal_success=bool(info['success'].item()),goal_progress=initial-goal_distance(env),
                             final_goal_distance=goal_distance(env),terminated=term,truncated=trunc))
    result=dict(status='PASS',episodes=episodes,transitions=steps,updates=0,planner_seconds=planner,
                success_rate=sum(e['success'] for e in episodes)/len(episodes),
                mean_return=sum(e['return_'] for e in episodes)/len(episodes),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    result['selection_used']=False if a.heldout else True
    result['model_sha256']=sha256(a.out/('best-model.pt' if a.heldout else 'evaluation-model.pt'))
    if a.eval_output.exists(): raise FileExistsError(a.eval_output)
    atomic_json(a.eval_output,result); env.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--preflight',type=Path); p.add_argument('--resume-check',type=Path)
    p.add_argument('--collector-check',type=Path); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--approve-budget',type=int); p.add_argument('--resume',action='store_true')
    p.add_argument('--acknowledge-rollback',action='store_true')
    p.add_argument('--eval-output',type=Path)
    p.add_argument('--heldout',action='store_true')
    a=p.parse_args()
    if a.eval_output: evaluate(a); return 0
    if a.approve_budget!=300000: p.error('explicit --approve-budget 300000 required; 24h/task and resource limits apply')
    if not all([a.preflight,a.resume_check,a.collector_check]): p.error('all three evidence paths required')
    import torch
    from tvwm.agents.upstream import load_api,verify_sources
    from tvwm.agents.resume import save_learner,load_learner
    from tvwm.agents.nominal import collect,code_hash,seed_all,empty_replay,atomic_json
    from tvwm.envs.maniskill import ManiSkillAdapter
    from tvwm.envs.contracts import compare_recorded_contract
    plan=read(ROOT/'configs/phase1/nominal.json'); evidence=read(a.preflight/'report.json')
    resume=read(a.resume_check/'report.json'); collector=read(a.collector_check/'report.json')
    if evidence['status']!='PASS_COMPATIBILITY_ONLY' or resume['status']!='PASS_LEARNER_RESUME_ONLY' or collector['status']!='PASS_COLLECTOR_BOUNDARY':
        raise ValueError('mandatory preflight gate failed')
    if not evidence['task']==resume['task']==collector['task']: raise ValueError('gate task mismatch')
    if not evidence['checkpoint_sha256']==resume['checkpoint_sha256']==collector['preflight_checkpoint_sha256']==sha256(a.preflight/'learner-envelope.pt'):
        raise ValueError('gate checkpoint mismatch')
    if collector['code_sha256']!=code_hash(): raise ValueError('code/config changed after collector check; rerun check')
    if verify_sources()!=evidence['sources']: raise ValueError('source mismatch')
    if {k:md.version(k) for k in evidence['packages']}!=evidence['packages']: raise ValueError('package mismatch')
    if sha256(ROOT/'environment/requirements-linux.lock')!=evidence['dependency_lock_sha256']: raise ValueError('lock mismatch')
    if a.resume:
        if not a.acknowledge_rollback: p.error('resume requires --acknowledge-rollback; uncheckpointed physical work may be repeated')
        if not (a.out/'latest.pt').is_file(): raise ValueError('missing latest checkpoint')
    else: a.out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); initial_resources=resources(a.out,plan)
    Agent,Buffer=load_api()
    from common.parser import cfg_to_dataclass
    from omegaconf import OmegaConf
    values=read(a.preflight/'resolved_config.json')
    if (values['batch_size'],values['model_size'],values['steps'],values['buffer_size'],values['seed_steps'])!=(256,5,300000,300000,1000):
        raise ValueError('resolved training configuration mismatch')
    cfg=cfg_to_dataclass(OmegaConf.create(values)); seed_all(plan['seed'])
    env=ManiSkillAdapter(evidence['task']); env.reset(seed=1)
    compare_recorded_contract(env.contract,evidence['contract'],list(evidence['contract']))
    # Constructor/reset happens before RNG restoration. Fresh training never loads smoke weights.
    agent=Agent(cfg); replay=Buffer(cfg)
    provenance=dict(task=evidence['task'],sources=evidence['sources'],config=vars(cfg),plan=plan,
                    code_sha256=code_hash(),packages=evidence['packages'],dependency_lock_sha256=evidence['dependency_lock_sha256'])
    counters=dict(transitions=0,updates=0,episode_index=0,evaluation_transitions=0,next_validation=0,
                  next_checkpoint=10000,best_score=None,best_model_file=None,best_model_sha256=None,
                  best_validation_file=None,best_envelope_file=None,wall_seconds=0.,boundary=True,replay_device=None)
    if a.resume:
        metadata=read(a.out/'latest.json')
        if sha256(a.out/'latest.pt')!=metadata['sha256']: raise ValueError('latest checkpoint hash mismatch')
        replay=empty_replay(Buffer,cfg,metadata['replay_device'])
        counters=load_learner(a.out/'latest.pt',agent,replay,provenance)
        if not counters['boundary'] or counters['transitions']>=300000: raise ValueError('checkpoint is not resumable unfinished boundary')
        prior=read(a.out/'progress.json') if (a.out/'progress.json').exists() else {}
        atomic_json(a.out/f'resume-{time.time_ns()}.json',dict(restored=counters,previous_observed_progress=prior,
                    rollback_acknowledged=True,warning='Physical work after last durable checkpoint can repeat; old metrics remain tagged by attempt.'))
        counters['wall_seconds']=max(counters['wall_seconds'],prior.get('wall_seconds',0.))
        if counters['best_model_file']:
            selected=a.out/counters['best_model_file']
            if sha256(selected)!=counters['best_model_sha256']: raise ValueError('selected model hash mismatch')
            shutil.copy2(selected,a.out/'best-model.pt')
            shutil.copy2(a.out/counters['best_validation_file'],a.out/'best-validation.json')
        # Restore aliases to the selection saved in this checkpoint, not a later rolled-back attempt.
        if counters['best_envelope_file']:
            chosen=a.out/counters['best_envelope_file']
            chosen_meta=read(chosen.with_suffix('.json'))
            if sha256(chosen)!=chosen_meta['sha256']: raise ValueError('selected envelope hash mismatch')
            shutil.copy2(chosen,a.out/'best.pt'); shutil.copy2(chosen.with_suffix('.json'),a.out/'best.json')
        elif (a.out/'best.pt').exists():
            (a.out/'best.pt').rename(a.out/f'orphan-best-{time.time_ns()}.pt')
            if (a.out/'best.json').exists(): (a.out/'best.json').rename(a.out/f'orphan-best-{time.time_ns()}.json')
    else:
        atomic_json(a.out/'resolved_config.json',vars(cfg)); atomic_json(a.out/'nominal_config.json',plan)
        atomic_json(a.out/'provenance.json',provenance)
    ledger_path=a.out/'wall-ledger.json'
    prior_wall=counters['wall_seconds']
    if a.resume and ledger_path.exists():
        ledger=read(ledger_path)
        prior_wall=max(prior_wall,ledger['charged_seconds']+(max(0.,time.time()-ledger['started_epoch']) if ledger['active'] else 0.))
    if prior_wall>=plan['max_wall_seconds_per_task']-900:
        raise RuntimeError('cumulative wall budget exhausted; no automatic extension')
    atomic_json(ledger_path,dict(charged_seconds=prior_wall,started_epoch=time.time(),active=True,
                                crash_accounting='Unclosed attempts conservatively include downtime until resume.'))
    atomic_json(a.out/'approval.json',dict(explicit_cli_budget=a.approve_budget,per_task_max_wall_seconds=86400,
                                         nominal_only=True,seed=1,model_size=5,batch_size=256))
    attempt=str(time.time_ns()); stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True)); signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    metrics=(a.out/f'metrics-{attempt}.jsonl').open('a',buffering=1)
    telemetry=(a.out/f'gpu-{attempt}.csv').open('w')
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu',
                              '--format=csv','-l','1'],stdout=telemetry)
    def elapsed(): return prior_wall+time.monotonic()-started
    def log(kind,data): metrics.write(json.dumps(dict(kind=kind,attempt=attempt,**data),allow_nan=False)+'\n')
    def update(data):
        log('update',data)
        if data['updates']%100==0: log('resources',resources(a.out,plan))
    def checkpoint(name):
        counters['wall_seconds']=elapsed(); counters['replay_device']=str(replay._storage_device)
        resources(a.out,plan)
        save_learner(a.out/name,agent,replay,counters,provenance)
        atomic_json(a.out/(Path(name).stem+'.json'),dict(sha256=sha256(a.out/name),replay_device=str(replay._storage_device),counters=dict(counters)))
    def validate():
        destination=a.out/f'validation-{counters["transitions"]:09d}-{attempt}.json'
        agent.save(a.out/'evaluation-model.pt')
        subprocess.run([sys.executable,__file__,'--out',str(a.out),'--eval-output',str(destination)],check=True,timeout=600)
        result=read(destination); counters['evaluation_transitions']+=result['transitions']
        result['training_transitions']=counters['transitions']; result['learner_updates']=counters['updates']
        result['checkpoint_model_sha256']=sha256(a.out/'evaluation-model.pt'); atomic_json(destination,result)
        log('validation',result)
        counters['next_validation']+=plan['validation_interval_transitions']
        score=[result['success_rate'],result['mean_return']]
        improved=counters['best_score'] is None or tuple(score)>tuple(counters['best_score'])
        if improved:
            counters['best_score']=score
            counters['best_model_file']=f'selected-model-{counters["transitions"]:09d}-{attempt}.pt'
            counters['best_model_sha256']=result['checkpoint_model_sha256']
            counters['best_validation_file']=destination.name
            counters['best_envelope_file']=(f'selected-envelope-{counters["transitions"]:09d}-{attempt}.pt' if replay.num_eps else None)
            shutil.copy2(a.out/'evaluation-model.pt',a.out/counters['best_model_file'])
            shutil.copy2(a.out/'evaluation-model.pt',a.out/'best-model.pt')
            atomic_json(a.out/'best-validation.json',result)
            if replay.num_eps:
                checkpoint(counters['best_envelope_file'])
                chosen=a.out/counters['best_envelope_file']
                shutil.copy2(chosen,a.out/'best.pt'); shutil.copy2(chosen.with_suffix('.json'),a.out/'best.json')
    status='FAIL'
    try:
        log('resources',initial_resources)
        if counters['next_validation']==0: validate()
        while counters['transitions']<plan['environment_transitions']:
            if stop[0] or elapsed()>=plan['max_wall_seconds_per_task']-900: break
            counters['boundary']=False
            result=collect(env,agent,replay,counters,plan['seed'],on_update=update)
            result.pop('trace'); counters['boundary']=result['complete']
            counters['wall_seconds']=elapsed()
            log('episode',dict(**result,**counters)); log('resources',resources(a.out,plan))
            atomic_json(a.out/'progress.json',dict(counters))
            if counters['transitions']%1000==0:
                print(json.dumps(dict(task=evidence['task'],**counters)),flush=True)
            if counters['transitions']>=counters['next_validation'] and (result['complete'] or counters['transitions']==300000): validate()
            if result['complete'] and not (a.out/'latest.pt').exists(): checkpoint('latest.pt')
            if result['complete'] and counters['transitions']>=counters['next_checkpoint']:
                counters['next_checkpoint']+=plan['checkpoint_interval_transitions']
                checkpoint(f'checkpoint-{counters["transitions"]:09d}.pt'); checkpoint('latest.pt')
            if not result['complete'] and counters['transitions']!=300000: raise AssertionError('unexpected partial episode')
        if counters['transitions']==300000:
            if counters['updates']!=299999: raise AssertionError('learner schedule mismatch')
            checkpoint('final.pt'); status='COMPLETE_NOMINAL_TRAINING'
            if counters['boundary']: checkpoint('latest.pt')
        else:
            if replay.num_eps: checkpoint('latest.pt')
            status='STOPPED_AT_EPISODE_BOUNDARY'
    except Exception as e:
        log('failure',dict(error=repr(e),traceback=traceback.format_exc()))
        # Never publish potentially half-updated state over the last durable checkpoint.
        raise
    finally:
        counters['wall_seconds']=elapsed()
        atomic_json(ledger_path,dict(charged_seconds=elapsed(),started_epoch=time.time(),active=False))
        atomic_json(a.out/f'report-{attempt}.json',dict(status=status,counters=counters,provenance=provenance,
                    initial_resources=initial_resources,competence='UNRESOLVED: inspect validation seed distribution; no Phase 2 authorization'))
        monitor.terminate(); monitor.wait(timeout=10); telemetry.close(); metrics.close(); env.close()
    print(json.dumps(dict(status=status,counters=counters),indent=2))
    return 0 if status=='COMPLETE_NOMINAL_TRAINING' else 2

if __name__=='__main__': sys.exit(main())
