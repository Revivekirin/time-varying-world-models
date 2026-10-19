"""Independent frozen Phase 0 checkpoint evaluation: two fixed held-out seeds only."""
import argparse
import json
from pathlib import Path
import random
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--smoke',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    evidence=json.loads((args.smoke/'report.json').read_text())
    if evidence['status']!='PASS' or evidence['stage']!='learner': raise ValueError('successful learner evidence required')
    args.out.mkdir(parents=True,exist_ok=False)
    def deadline(*_): raise TimeoutError('evaluation timeout')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(900)
    import numpy as np
    import torch
    from tvwm.agents.upstream import load_api, config, verify_sources
    from tvwm.envs.maniskill import ManiSkillAdapter
    if verify_sources()!=evidence['sources']: raise ValueError('source mismatch')
    Agent,_=load_api()
    env=ManiSkillAdapter(evidence['task'])
    report=dict(status='RUNNING',task=evidence['task'],episodes=[],transitions=0,updates=0)
    start=time.perf_counter()
    try:
        obs,_=env.reset(seed=10001)
        agent=Agent(config(evidence['task'],obs.numel(),env.action_dim,env.horizon))
        agent.load(args.smoke/'agent.pt')
        agent.model.eval()
        agent.model.requires_grad_(False)
        for seed in [10001,10002]:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            obs,_=env.reset(seed=seed);ret=0;any_success=False
            for t in range(env.horizon):
                action=agent.act(obs,t0=t==0,eval_mode=True)
                obs,r,term,trunc,info=env.step(action)
                report['transitions']+=1;ret+=float(r);any_success|=bool(info['success'].item())
                if term or trunc: break
            else: raise AssertionError('missing boundary')
            report['episodes'].append(dict(seed=seed,length=t+1,return_=ret,success=any_success,terminal_success=bool(info['success'].item()),terminated=term,truncated=trunc))
        report['status']='PASS'
    except Exception as e:
        report.update(status='FAIL',error=repr(e));raise
    finally:
        report['wall_seconds']=time.perf_counter()-start
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        env.close();signal.alarm(0)

if __name__=='__main__':main()
