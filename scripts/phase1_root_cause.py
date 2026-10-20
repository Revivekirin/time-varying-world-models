"""User-run bounded diagnostics. Never trains, modifies upstream, or grants launch readiness."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def flatten(value, prefix=''):
    if isinstance(value,dict):
        result={}
        for k,v in value.items(): result.update(flatten(v,prefix+'/'+str(k)))
        return result
    if isinstance(value,(list,tuple)):
        result={}
        for k,v in enumerate(value): result.update(flatten(v,prefix+'/'+str(k)))
        return result
    return {prefix:value}


def compare(left,right):
    import torch
    result={}; exact=True; close=True
    for key in sorted(left.keys()|right.keys()):
        if key not in left or key not in right:
            result[key]={'missing':True}; exact=close=False; continue
        a,b=left[key],right[key]
        if torch.is_tensor(a) and torch.is_tensor(b):
            a=a.cpu();b=b.cpu()
            if a.shape!=b.shape:
                result[key]={'shapes':[list(a.shape),list(b.shape)],'close':False};exact=close=False;continue
            delta=(a.double()-b.double()).abs()
            eq=a==b; within=torch.isclose(a.double(),b.double(),rtol=0,atol=1e-6)
            nz=(~eq).nonzero(); bad=(~within).nonzero()
            d={'exact':bool(eq.all()),'close':bool(within.all()),'max_abs_difference':float(delta.max()) if delta.numel() else 0.,
               'first_nonzero_index':nz[0].tolist() if len(nz) else None,
               'first_outside_tolerance_index':bad[0].tolist() if len(bad) else None}
            if len(bad):
                index=tuple(bad[0].tolist()); d.update(reference=float(a[index]),child=float(b[index]))
        else: d={'exact':a==b,'close':a==b,'reference':a,'child':b}
        exact &= d['exact']; close &= d['close']; result[key]=d
    return dict(exact=exact,within_tolerance=close,fields=result)


def snapshot(env,obs,missing):
    import torch
    state={'obs':obs.detach().cpu().clone()}
    def capture(name,fn):
        try:
            for key,value in flatten(fn(),name).items():
                if hasattr(value,'raw_pose'): state[key]=value.raw_pose.detach().cpu().clone()
                elif torch.is_tensor(value): state[key]=value.detach().cpu().clone()
                elif isinstance(value,(int,float,bool,str)) or value is None: state[key]=value
                else: missing.add(key+': unsupported '+type(value).__name__)
        except Exception as e: missing.add(name+': '+repr(e))
    robot=env.base.agent.robot; obj=env.base.obj
    capture('robot/qpos',robot.get_qpos);capture('robot/qvel',robot.get_qvel)
    capture('robot/root_pose',lambda:robot.pose.raw_pose)
    capture('robot/root_linear_velocity',lambda:robot.root_linear_velocity)
    capture('robot/root_angular_velocity',lambda:robot.root_angular_velocity)
    capture('object/pose',lambda:obj.pose.raw_pose)
    capture('object/linear_velocity',lambda:obj.linear_velocity)
    capture('object/angular_velocity',lambda:obj.angular_velocity)
    capture('agent/state',env.base.agent.get_state)
    capture('controller/state',env.base.agent.controller.get_state)
    for name,controller in env.base.agent.controller.controllers.items():
        for field in ['_target_qpos','_start_qpos','_step','_step_size']:
            if field=='_step_size' and not controller.config.interpolate: continue
            capture('controller/'+name+'/'+field,lambda c=controller,f=field:getattr(c,f))
    return state


def state_digest(state):
    """Canonical typed hash of tensors and state_dict metadata; no pickle addresses."""
    import torch
    h=hashlib.sha256()
    def token(data):
        h.update(len(data).to_bytes(8,'big'));h.update(data)
    def visit(value):
        token((type(value).__module__+'.'+type(value).__qualname__).encode())
        if torch.is_tensor(value):
            token(str(value.dtype).encode());visit(tuple(value.shape))
            token(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value,dict):
            if not all(isinstance(key,str) for key in value): raise TypeError('state keys must be strings')
            for key in sorted(value): visit(key);visit(value[key])
        elif isinstance(value,(tuple,list)):
            token(str(len(value)).encode())
            for item in value: visit(item)
        elif value is None or isinstance(value,(str,int,float,bool)):
            token(repr(value).encode())
        elif isinstance(value,(torch.device,torch.dtype)):
            token(str(value).encode())
        else: raise TypeError('unsupported checkpoint metadata: '+str(type(value)))
    visit(state)
    return h.hexdigest()


def model_digest(agent):
    return state_digest(agent.model.state_dict())


def worker(a):
    import torch
    import importlib.metadata as md
    from tvwm.agents.resume import restore_rng,rng_state,load_learner
    from tvwm.agents.upstream import verify_sources,load_api
    from tvwm.agents.nominal import training_seed,empty_replay
    checkpoint=torch.load(a.source/'boundary.pt',weights_only=False)
    provenance=checkpoint['provenance']
    if provenance['task']!='PushCube-v1' or verify_sources()!=provenance['sources']: raise ValueError('task/source mismatch')
    if {k:md.version(k) for k in provenance['packages']}!=provenance['packages']: raise ValueError('package mismatch')
    if digest(ROOT/'environment/requirements-linux.lock')!=provenance['dependency_lock_sha256']: raise ValueError('dependency lock mismatch')
    trace=torch.load(a.source/('child.pt' if a.worker=='planner-child' else 'reference.pt'),weights_only=False)['trace']
    if not 0<len(trace['actions'])<=50: raise ValueError('trace budget')
    output={'mode':a.worker,'checkpoint_sha256':digest(a.source/'boundary.pt'),'records':[],'missing':[]}
    if a.worker.startswith('env'):
        from tvwm.envs.maniskill import ManiSkillAdapter
        env=ManiSkillAdapter('PushCube-v1');env.reset(seed=1);missing=set()
        if a.worker in ('env-warm','env-warm-probe'):
            history=json.loads(a.warm_history.read_text())
            if history.get('original_recording') is not True: raise ValueError('original recorded warm history required')
            if history.get('boundary_sha256')!=digest(a.source/'boundary.pt'): raise ValueError('warm history must bind original boundary checkpoint')
            episodes=history['episodes']
            # Original collector performed exactly one warm episode between initial load and boundary.
            if len(episodes)!=1: raise ValueError('expected original single warm episode')
            for episode in episodes:
                path=(a.warm_history.parent/episode['path']).resolve()
                if digest(path)!=episode['sha256']: raise ValueError('warm trace hash mismatch')
                expected=training_seed(1,checkpoint['counters']['episode_index']-1)
                if episode['seed']!=expected: raise ValueError('warm seed mismatch')
                warm=torch.load(path,weights_only=False)['trace']
                if not 0<len(warm['actions'])<=50: raise ValueError('warm budget')
                env.reset(seed=episode['seed'])
                for index,action in enumerate(warm['actions']):
                    _,_,term,trunc,_=env.step(action)
                    if (term or trunc)!=(index==len(warm['actions'])-1): raise ValueError('warm boundary mismatch')
            output['warm_history_sha256']=digest(a.warm_history)
        # No simulator state exists in this learner envelope; restore available RNG only.
        restore_rng(checkpoint['rng'])
        seed=training_seed(1,checkpoint['counters']['episode_index'])
        obs,_=env.reset(seed=seed)
        output.update(seed=seed,contract=env.contract,sim_config=env.base.sim_config.dict(),
                      enhanced_determinism=env.base._enhanced_determinism,reconfiguration_freq=env.base.reconfiguration_freq,restored_state='RNG only; checkpoint has no simulator state')
        output['records'].append(snapshot(env,obs,missing))
        probe=[]
        if a.worker in ('env-probe','env-warm-probe'):
            from tvwm.envs.maniskill import observation
            original_hook=env.base._after_simulation_step
            substep=[0]
            def probe_hook():
                original_hook(); substep[0]+=1
                if substep[0]==1 or a.first_control_only:
                    env.base.scene._gpu_fetch_all()
                    probe.append(snapshot(env,observation(env.base.get_obs()),missing))
            env.base._after_simulation_step=probe_hook
        actions=trace['actions'][:1] if a.first_control_only else trace['actions']
        for index,action in enumerate(actions):
            if a.worker in ('env-probe','env-warm-probe'): substep[0]=0
            obs,_,term,trunc,_=env.step(action)
            record=snapshot(env,obs,missing);record['applied_action']=action.cpu().clone();output['records'].append(record)
            if (term or trunc)!=(index==len(trace['actions'])-1): raise ValueError('target boundary mismatch')
        if a.worker in ('env-probe','env-warm-probe'):
            output['first_substep_records']=probe
            output['probe_limitation']='Additional GPU fetch and host reads may perturb execution; compare probe control trajectory against native env-a.'
        output['missing']=sorted(missing)+['PhysX contact/solver caches and hidden GPU state unavailable.']
        output['instrumentation']=('Separate first-substep GPU fetch probe enabled; production unchanged.' if a.worker in ('env-probe','env-warm-probe') else
            'Native GPU fetch occurs after five physics substeps; control-step records only, no stepping hooks.')
        env.close()
    else:
        Agent,Buffer=load_api()
        from omegaconf import OmegaConf
        from common.parser import cfg_to_dataclass
        cfg=cfg_to_dataclass(OmegaConf.create(provenance['config']))
        agent=Agent(cfg);output['rng_after_construction']=rng_state()
        replay=empty_replay(Buffer,cfg,'cuda:0')
        counters=load_learner(a.source/'boundary.pt',agent,replay,provenance)
        output['rng_after_restore']=rng_state();output['model_sha256']=model_digest(agent)
        output['restored_rng_matches_checkpoint']=compare(
            flatten({'cpu':checkpoint['rng']['cpu'],'cuda':checkpoint['rng']['cuda']}),
            flatten({'cpu':output['rng_after_restore']['cpu'],'cuda':output['rng_after_restore']['cuda']}))
        output['counters']=counters
        for index,obs in enumerate(trace['obs'][:-1]):
            before=rng_state();mean_before=agent._prev_mean.detach().cpu().clone()
            action=agent.act(obs,t0=(index==0 or a.worker=='planner-all-t0'),eval_mode=False)
            after=rng_state()
            output['records'].append({'obs':obs.cpu(),'action':action.cpu(),'mean_before':mean_before,
                'mean_after':agent._prev_mean.detach().cpu().clone(),'cpu_rng_before':before['cpu'],
                'cpu_rng_after':after['cpu'],'cuda_rng_before':before['cuda'],'cuda_rng_after':after['cuda'],
                't0':index==0 or a.worker=='planner-all-t0'})
        output['model_sha256_after']=model_digest(agent)
        output['missing']=['Candidate/elite tensors are local variables in upstream plan; no upstream patch or extra sampling used.',
                           'Environment reset is intentionally absent; these sequences isolate planner behavior from identical restored RNG.']
    torch.save(output,a.destination)


def source_refs():
    requests={
        'src/tvwm/envs/maniskill.py':['def reset','def step'],
        'src/tvwm/agents/nominal.py':['def training_seed','def collect'],
        'src/tvwm/agents/resume.py':['def save_learner','def load_learner','restore_rng(state'],
        'external/ManiSkill/mani_skill/envs/sapien_env.py':['def reset','controller.reset()','self.scene.step()','self.scene._gpu_fetch_all()'],
        'external/ManiSkill/mani_skill/agents/controllers/pd_joint_pos.py':['def reset','def get_state','def set_action'],
        'external/tdmpc2/tdmpc2/tdmpc2.py':['def act','if not t0:','torch.topk','gumbel_softmax_sample','self._prev_mean.copy_']}
    return {path:[{'line':i,'text':line.strip()} for i,line in enumerate((ROOT/path).read_text().splitlines(),1)
                   if any(needle in line for needle in needles)] for path,needles in requests.items()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path)
    p.add_argument('--warm-history',type=Path)
    p.add_argument('--planner-only',action='store_true')
    p.add_argument('--warm-comparison-only',action='store_true')
    p.add_argument('--first-control-only',action='store_true')
    p.add_argument('--worker',choices=['env-fresh','env-warm','env-probe','env-warm-probe','planner-ref','planner-child','planner-all-t0'])
    p.add_argument('--destination',type=Path)
    a=p.parse_args()
    if a.worker: worker(a);return
    if a.out is None: p.error('--out required')
    if a.planner_only and a.warm_history: p.error('--planner-only cannot include warm history')
    if a.warm_comparison_only and (a.planner_only or not a.warm_history): p.error('--warm-comparison-only requires warm history and excludes planner-only')
    if a.first_control_only and not a.warm_comparison_only: p.error('--first-control-only requires --warm-comparison-only')
    a.out.mkdir(parents=True,exist_ok=False)
    required=['reference.pt','child.pt','boundary.pt']
    hashes={name:digest(a.source/name) for name in required}
    write(a.out/'input-hashes.json',hashes);write(a.out/'source-references.json',source_refs())
    report={'phase1_readiness':'NO-GO','optimizer_updates':0,'training_transitions':0,
            'max_simulation_transitions':0 if a.planner_only else (300 if a.warm_history else 200),'max_planner_calls':250,'comparisons':{}}
    jobs=[('env-fresh','env-a'),('env-fresh','env-b'),('env-probe','probe-a'),('env-probe','probe-b'),('planner-ref','planner-a'),('planner-ref','planner-b'),
          ('planner-child','planner-child'),('planner-all-t0','planner-t0-a'),('planner-all-t0','planner-t0-b')]
    if a.planner_only: jobs=[job for job in jobs if job[0].startswith('planner')]
    if a.warm_history:jobs.append(('env-warm','env-warm'))
    if a.warm_comparison_only:
        jobs=[('env-fresh','env-b'),('env-warm','env-warm')]
        report['max_simulation_transitions']=150;report['max_planner_calls']=0
        if a.first_control_only:
            jobs += [('env-probe','probe-b'),('env-warm-probe','probe-warm')]
            report['max_simulation_transitions']=104
    try:
        import torch
        report['worker_errors']={}
        for mode,name in jobs:
            command=[sys.executable,__file__,'--source',str(a.source),'--worker',mode,'--destination',str(a.out/(name+'.pt'))]
            if a.first_control_only:command+=['--first-control-only']
            if a.warm_history:command+=['--warm-history',str(a.warm_history)]
            try:
                with (a.out/(name+'.log')).open('w') as log:
                    subprocess.run(command,check=True,timeout=180,stdout=log,stderr=subprocess.STDOUT)
                print('Completed '+name,flush=True)
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
                report['worker_errors'][name]=repr(error)
                print('Failed '+name+'; see worker log. Continuing independent diagnostics.',flush=True)
        pairs=[('A1','env-a','env-b'),('A1_probe','probe-a','probe-b'),('probe_effect','env-a','probe-a'),('B1','planner-a','planner-b'),('B1_all_t0','planner-t0-a','planner-t0-b'),('B2','planner-a','planner-child')]
        if a.planner_only:
            pairs=[pair for pair in pairs if pair[0].startswith('B')]
            report['comparisons']['A1']={'status':'NOT_RERUN','reason':'planner-only; previous environment evidence remains separate'}
        if a.warm_history:pairs.append(('A2','env-warm','env-b'))
        else:report['comparisons']['A2']={'status':'UNRESOLVED','reason':'Original recorded warm episode actions unavailable; no replacement history generated.'}
        if a.warm_comparison_only: pairs=[pair for pair in pairs if pair[0]=='A2']
        if a.first_control_only:
            pairs += [('A2_probe','probe-warm','probe-b'),('warm_probe_effect','env-warm','probe-warm'),('fresh_probe_effect','env-b','probe-b')]
        for label,l,r in pairs:
            if l in report['worker_errors'] or r in report['worker_errors']:
                report['comparisons'][label]={'status':'UNRESOLVED','reason':'worker error; see worker_errors and logs'}
                continue
            left=torch.load(a.out/(l+'.pt'),weights_only=False);right=torch.load(a.out/(r+'.pt'),weights_only=False)
            steps=[compare(flatten(x),flatten(y)) for x,y in zip(left['records'],right['records'])]
            equal_lengths=len(left['records'])==len(right['records'])
            detail={'steps':steps,'same_record_count':equal_lengths,'missing':left['missing']+right['missing']}
            if label in ('A1_probe','A2_probe'):
                detail['first_substep_comparisons']=[compare(flatten(x),flatten(y)) for x,y in zip(left['first_substep_records'],right['first_substep_records'])]
            if label.startswith('B'):
                detail['same_model']=left['model_sha256']==right['model_sha256']==left['model_sha256_after']==right['model_sha256_after']
                detail['restored_rng_left']=left['restored_rng_matches_checkpoint'];detail['restored_rng_right']=right['restored_rng_matches_checkpoint']
                detail['rng_after_construction']=compare(flatten({k:left['rng_after_construction'][k] for k in ['cpu','cuda']}),flatten({k:right['rng_after_construction'][k] for k in ['cpu','cuda']}))
                detail['action_first_outside_tolerance']=next((i for i,s in enumerate(steps) if not s['fields']['/action']['close']),None)
            detail['first_nonexact_record']=next((i for i,s in enumerate(steps) if not s['exact']),None)
            detail['first_outside_tolerance_record']=next((i for i,s in enumerate(steps) if not s['within_tolerance']),None)
            passed=equal_lengths and all(s['within_tolerance'] for s in steps)
            if label=='A2_probe':
                substeps=detail['first_substep_comparisons']
                detail['first_divergent_physics_substep']=next((i+1 for i,item in enumerate(substeps) if not item['within_tolerance']),None)
                passed &= len(left['first_substep_records'])==len(right['first_substep_records'])==5 and all(item['within_tolerance'] for item in substeps)
            if label.startswith('B'):passed &= detail['same_model'] and detail['restored_rng_left']['exact'] and detail['restored_rng_right']['exact']
            detail['status']=('SENSITIVITY_DIAGNOSTIC' if label=='B2' else ('PASS' if passed else 'FAIL'))
            write(a.out/(label+'-per-step.json'),detail)
            report['comparisons'][label]={k:v for k,v in detail.items() if k not in ['steps','first_substep_comparisons','rng_after_construction','missing']}
        report['input_hashes_unchanged']=hashes=={name:digest(a.source/name) for name in required}
        report['status']='DIAGNOSTICS_COMPLETED' if not report['worker_errors'] else 'DIAGNOSTICS_PARTIAL'
    except Exception as e:report.update(status='DIAGNOSTIC_ERROR',error=repr(e),traceback=traceback.format_exc())
    write(a.out/'summary.json',report)
    lines=['# Collector boundary root-cause diagnostic report','',
           'Phase 1 readiness: **NO-GO**. This diagnostic does not authorize training.',
           '', 'VERIFIED: values below are measured by these diagnostic processes; original protocol/tolerance unchanged.',
           'Classification: UNRESOLVED pending cross-experiment interpretation. A failed comparison alone does not establish causality.', '']
    for label,value in report['comparisons'].items():lines += ['## '+label,'', '```json',json.dumps(value,indent=2),'```','']
    lines += ['## Limitations and causal interpretation','',
        'A1 compares fresh environments using fixed actions. A2 requires original warm history and is unavailable without it.',
        'Native adapter observations occur after five physics substeps. A1_probe records first substeps via an additional GPU fetch/read hook; probe_effect checks its effect on control-boundary trajectories. Hidden solver caches remain inaccessible.',
        'B1 isolates planner with identical restored RNG and observations; no simulator reset is run. Construction RNG may differ; restored RNG must match.',
        'B2 action differences with matched initial state support sensitivity to changed observations, not a general numerical-instability claim.',
        'PASS A1 / FAIL A2 / PASS B1 prioritizes residual history. FAIL A1 prioritizes environment reproducibility. FAIL B1 prioritizes planner state/RNG. These are investigative priorities only.',
        'No production or upstream fix is justified automatically. No tolerances are relaxed.',
        'Rerun original Push and Pick collector checks only after an evidence-backed fix; both must pass rtol=0, atol=1e-6 before reconsidering readiness.',
        '', 'Exact path/line evidence: source-references.json. Per-step physical/controller and RNG/planner comparisons: *-per-step.json. Raw state and RNG tensors: worker .pt files.']
    (a.out/'root-cause-report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2))
    if report['status']!='DIAGNOSTICS_COMPLETED':sys.exit(1)

if __name__=='__main__':main()
