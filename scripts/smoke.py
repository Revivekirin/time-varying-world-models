"""Bounded Phase 0 only. User executes on Linux CUDA after dependency review."""
import argparse
import json
import math
from pathlib import Path
import random
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=['PushCube-v1','PickCube-v1'])
    parser.add_argument('--stage', required=True, choices=['contract','learner'])
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--contract', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(task=args.task, stage=args.stage, status='RUNNING', transitions=0,
                  resets=0, updates=0, tests={}, seed=1)
    started = time.perf_counter()
    env = None
    def deadline(*_): raise TimeoutError('bounded Phase 0 timeout')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(300 if args.stage == 'contract' else 900)
    try:
        import numpy as np
        import torch
        from tvwm.envs.maniskill import ManiSkillAdapter
        from tvwm.agents.upstream import verify_sources, load_api, config, episode_tensor
        from tvwm.utils.state import snapshot_state, assert_state_equal
        report['sources'] = verify_sources()
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU substitution for upstream learner')
        # Verify imported ManiSkill is exactly the audited editable source.
        import mani_skill
        if not Path(mani_skill.__file__).resolve().is_relative_to((ROOT/'external/ManiSkill').resolve()):
            raise RuntimeError('install audited ManiSkill checkout with pip install -e external/ManiSkill')
        if args.stage == 'learner':
            if args.contract is None:
                raise ValueError('--contract path to successful contract report required')
            previous = json.loads(args.contract.read_text())
            if (previous['status'],previous['task'],previous['stage'],previous['sources']) != ('PASS',args.task,'contract',report['sources']):
                raise ValueError('contract evidence mismatch')
        random.seed(1); np.random.seed(1); torch.manual_seed(1); torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.benchmark = False
        report['determinism'] = 'Seeds recorded; GPU physics bitwise determinism not guaranteed'
        torch.cuda.reset_peak_memory_stats()
        env = ManiSkillAdapter(args.task)
        def reset(seed):
            report['resets'] += 1
            return env.reset(seed)[0]
        def step(action):
            result = env.step(action)
            report['transitions'] += 1
            report['physics_substeps'] = report['transitions'] * env.base.sim_freq // env.base.control_freq
            return result
        def timed(fn):
            torch.cuda.synchronize(); start=time.perf_counter()
            value=fn()
            torch.cuda.synchronize()
            return value,time.perf_counter()-start
        if args.stage == 'contract':
            a = reset(123); b = reset(123)
            report['reset_max_abs_difference'] = float((a-b).abs().max())
            torch.testing.assert_close(a,b,rtol=0,atol=1e-6)
            report['tests']['reset_reproducibility'] = 'PASS for initial state, tolerance 1e-6'
            raw = env.base.get_obs(unflattened=True)
            def shapes(x):
                return {k:shapes(v) for k,v in x.items()} if isinstance(x,dict) else {'shape':list(x.shape),'dtype':str(x.dtype)}
            report['observation_structure'] = shapes(raw)
            # Public state roundtrip at a control boundary. Not a full RNG/time restoration claim.
            import copy
            state = copy.deepcopy(env.base.get_state_dict())
            env.base.set_state_dict(state)
            env.base.agent.set_controller_state(copy.deepcopy(env.base.agent.get_controller_state()))
            from tvwm.envs.maniskill import observation
            torch.testing.assert_close(observation(env.base.get_obs()),b,rtol=0,atol=1e-6)
            report['tests']['state_identity_roundtrip'] = 'PASS; future transition restoration remains UNRESOLVED'
            episodes=[]; seconds=0
            for trial in range(2):
                reset(200+trial)
                for t in range(env.horizon):
                    action = torch.zeros(env.action_dim)
                    action[-1] = 1.0
                    if trial == 1: action[t % 7] = 0.1 * (-1 if t % 2 else 1)
                    (obs,reward,term,trunc,info),dt = timed(lambda:step(action))
                    seconds += dt
                    if term or trunc:
                        episodes.append(dict(length=t+1,terminated=term,truncated=trunc,success=bool(info['success'].item())))
                        break
                else: raise AssertionError('native horizon did not terminate/truncate')
            report['episodes']=episodes
            report['simulator_transitions_per_second']=report['transitions']/seconds
            report['tests']['A_B']='PASS'
        else:
            Agent, Buffer = load_api()
            obs = reset(1)
            cfg = config(args.task,obs.numel(),env.action_dim,env.horizon)
            (args.out/'resolved_config.json').write_text(json.dumps(vars(cfg),indent=2,default=str))
            agent = Agent(cfg)
            # Synthetic independent sampler test verifies actual upstream sequence indexing.
            synthetic = Buffer(cfg)
            for offset in (0,100):
                observations=[torch.full((obs.numel(),),float(offset+i)) for i in range(9)]
                actions=[torch.full((env.action_dim,),float(offset+i)) for i in range(8)]
                synthetic.add(episode_tensor(observations,actions,list(range(offset,offset+8)),[False]*8,[False]*7+[True]))
            for _ in range(4):
                so,sa,sr,st,_=synthetic.sample()
                torch.testing.assert_close(so[1:,:,0]-so[:-1,:,0],torch.ones_like(sr[...,0]))
                torch.testing.assert_close(sa[...,0],so[:-1,:,0])
                torch.testing.assert_close(sr[...,0],so[:-1,:,0])
                assert torch.isfinite(sa).all() and torch.isfinite(sr).all() and not st.any()
            del synthetic
            report['tests']['D']='PASS synthetic upstream sampler alignment, boundaries, sentinel exclusion'
            with torch.no_grad():
                z=agent.model.encode(obs[None].cuda(),None)
                pa,_=agent.model.pi(z,None)
                tensors=[z,pa,agent.model.next(z,pa,None),agent.model.reward(z,pa,None),agent.model.Q(z,pa,None),agent.model.termination(z,None)]
                assert all(torch.isfinite(x).all() for x in tensors)
            action,latency=timed(lambda:agent.act(obs,t0=True,eval_mode=True))
            assert action.shape==(env.action_dim,) and action.abs().max()<=1
            report['tests']['C']='PASS'; report['planning_seconds']=[latency]
            replay=Buffer(cfg)
            metrics=[]; update_times=[]
            report["losses"]=metrics
            report["update_seconds"]=update_times
            def update():
                values,dt=timed(lambda:agent.update(replay))
                report['updates']+=1
                row={k:float(v) for k,v in values.items()}
                if not all(math.isfinite(v) for v in row.values()): raise FloatingPointError(row)
                metrics.append(row);update_times.append(dt)
                report["learner_updates_per_second"]=len(update_times)/sum(update_times)
            # Four seed episodes, then exactly four learner updates.
            for ep in range(4):
                obs=reset(10+ep); observations=[obs];actions=[];rewards=[];terms=[];truncs=[]
                for t in range(env.horizon):
                    action=torch.rand(env.action_dim)*2-1
                    obs,r,term,trunc,_=step(action)
                    observations.append(obs);actions.append(action);rewards.append(r);terms.append(term);truncs.append(trunc)
                    if term or trunc: break
                if len(actions)<cfg.horizon: raise RuntimeError('episode too short for sequence; stop, do not resample indefinitely')
                replay.add(episode_tensor(observations,actions,rewards,terms,truncs))
            for _ in range(4): update()
            report['tests']['E_F']='PASS finite model/critic/actor metrics and gradient norms; 4 updates'
            # One MPPI episode, <=4 additional updates from completed seed episodes.
            obs=reset(90); observations=[obs];actions=[];rewards=[];terms=[];truncs=[]
            for t in range(env.horizon):
                action,dt=timed(lambda:agent.act(obs,t0=t==0))
                report['planning_seconds'].append(dt)
                obs,r,term,trunc,_=step(action)
                observations.append(obs);actions.append(action);rewards.append(r);terms.append(term);truncs.append(trunc)
                if t<4: update()
                if term or trunc: break
            if len(actions)>=cfg.horizon: replay.add(episode_tensor(observations,actions,rewards,terms,truncs))
            report['tests']['G']='PASS bounded planned collection and learner interleave'
            # Independently check target Polyak operation, then restore original state.
            before=snapshot_state(agent.model.state_dict())
            target=agent.model._target_Qs_params.data.clone()
            online=agent.model._detach_Qs_params.data.clone()
            agent.model.soft_update_target_Q()
            expected=target.clone()
            expected.lerp_(online,cfg.tau)
            for k,v in expected.items(True,True):
                torch.testing.assert_close(agent.model._target_Qs_params.data[k],v)
            agent.model.load_state_dict(before)
            report['tests']['target_update']='PASS explicit Polyak interpolation'
            checkpoint=args.out/'agent.pt';agent.save(checkpoint)
            with torch.no_grad():
                reference=agent.model.encode(obs[None].cuda(),None).clone()
                next(agent.model._encoder.parameters()).add_(1)
            agent.load(checkpoint)
            assert_state_equal(agent.model.state_dict(),before)
            with torch.no_grad(): torch.testing.assert_close(agent.model.encode(obs[None].cuda(),None),reference,rtol=0,atol=0)
            frozen=snapshot_state(agent.model.state_dict())
            rng=torch.cuda.get_rng_state_all();cpu_rng=torch.get_rng_state()
            first=agent.act(obs,t0=True,eval_mode=True)
            torch.cuda.set_rng_state_all(rng);torch.set_rng_state(cpu_rng)
            second=agent.act(obs,t0=True,eval_mode=True)
            torch.testing.assert_close(first,second,rtol=0,atol=0)
            assert_state_equal(agent.model.state_dict(),frozen)
            torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),
                            numpy=np.random.get_state(),python=random.getstate()),args.out/'rng.pt')
            report['tests']['H']='PASS checkpoint encoder restoration, seeded frozen MPPI; not resumable training'
            report['learner_updates_per_second']=len(update_times)/sum(update_times)
            report['losses']=metrics
            report['tests']['I']='MEASURED process timings; device-wide memory must also use nvidia-smi'
        report['contract']=env.contract
        report['torch_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        report['torch_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        report['gpu']=torch.cuda.get_device_name(0)
        report['status']='PASS'
    except Exception as error:
        report['status']='FAIL';report['error']=repr(error);report['traceback']=traceback.format_exc()
    finally:
        if env is not None: report['contract']=env.contract
        report['wall_seconds']=time.perf_counter()-started
        (args.out/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
        if env is not None: env.close()
        signal.alarm(0)
    print(json.dumps(report,indent=2,default=str))
    return 0 if report['status']=='PASS' else 1

if __name__=='__main__': sys.exit(main())
