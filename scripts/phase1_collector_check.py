"""Bounded fresh-process episode-boundary continuation test, no optimizer updates."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from phase1_resume_check import sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--preflight',type=Path,required=True); p.add_argument('--resume-check',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--child',action='store_true')
    a=p.parse_args(); start=time.monotonic()
    if not a.child: a.out.mkdir(parents=True,exist_ok=False)
    report=dict(status='FAIL',optimizer_updates=0,maximum_total_transitions=250)
    try:
        import torch
        import importlib.metadata as md
        from tvwm.agents.upstream import load_api,verify_sources
        from tvwm.agents.resume import load_learner,save_learner
        from tvwm.agents.nominal import collect,empty_replay,code_hash,atomic_json
        from tvwm.envs.maniskill import ManiSkillAdapter
        evidence=json.loads((a.preflight/'report.json').read_text())
        resume=json.loads((a.resume_check/'report.json').read_text())
        assert evidence['status']=='PASS_COMPATIBILITY_ONLY'
        assert resume['status']=='PASS_LEARNER_RESUME_ONLY' and resume['task']==evidence['task']
        assert resume['checkpoint_sha256']==evidence['checkpoint_sha256']==sha256(a.preflight/'learner-envelope.pt')
        assert verify_sources()==evidence['sources']
        assert {k:md.version(k) for k in evidence['packages']}==evidence['packages']
        assert sha256(ROOT/'environment/requirements-linux.lock')==evidence['dependency_lock_sha256']
        Agent,Buffer=load_api()
        from common.parser import cfg_to_dataclass
        from omegaconf import OmegaConf
        cfg=cfg_to_dataclass(OmegaConf.create(json.loads((a.preflight/'resolved_config.json').read_text())))
        env=ManiSkillAdapter(evidence['task']); env.reset(seed=1); agent=Agent(cfg)
        replay=empty_replay(Buffer,cfg,evidence['replay_storage_device'])
        provenance=dict(sources=evidence['sources'],config=vars(cfg),task=evidence['task'],
                        dependency_lock_sha256=evidence['dependency_lock_sha256'],packages=evidence['packages'])
        if a.child:
            counters=load_learner(a.out/'boundary.pt',agent,replay,provenance)
            episode=collect(env,agent,replay,counters,1,learn=False)
            torch.save(dict(trace=episode['trace'],counters=counters),a.out/'child.pt')
            env.close(); return 0
        counters=load_learner(a.preflight/'learner-envelope.pt',agent,replay,provenance)
        warm=collect(env,agent,replay,counters,1,learn=False)
        assert warm['complete']
        save_learner(a.out/'boundary.pt',agent,replay,counters,provenance)
        reference=collect(env,agent,replay,counters,1,learn=False)
        agent.save(a.out/'evaluation-model.pt')
        atomic_json(a.out/'resolved_config.json',vars(cfg))
        plan=json.loads((ROOT/'configs/phase1/nominal.json').read_text())
        plan['validation_seeds']=plan['validation_seeds'][:2]
        atomic_json(a.out/'nominal_config.json',plan)
        env.close()
        # Release parent CUDA allocations before starting the independent simulator.
        ref=reference['trace']; expected=dict(counters)
        del agent,replay,env; torch.cuda.empty_cache()
        subprocess.run([sys.executable,__file__,'--preflight',str(a.preflight),'--resume-check',str(a.resume_check),
                        '--out',str(a.out),'--child'],check=True,timeout=180)
        child=torch.load(a.out/'child.pt',weights_only=False)
        assert child['counters']==expected
        differences={}
        for key,value in ref.items():
            actual=child['trace'][key]
            torch.testing.assert_close(actual,value,rtol=0,atol=1e-6)
            differences[key]=float((actual.float()-value.float()).abs().max())
        subprocess.run([sys.executable,str(ROOT/'scripts/phase1_train.py'),'--out',str(a.out),
                        '--eval-output',str(a.out/'evaluation-check.json')],check=True,timeout=180)
        evaluation=json.loads((a.out/'evaluation-check.json').read_text())
        assert evaluation['status']=='PASS' and evaluation['updates']==0 and evaluation['transitions']<=100
        assert [e['seed'] for e in evaluation['episodes']]==plan['validation_seeds']
        assert evaluation['model_sha256']==sha256(a.out/'evaluation-model.pt')
        report.update(status='PASS_COLLECTOR_BOUNDARY',task=evidence['task'],code_sha256=code_hash(),
                      preflight_checkpoint_sha256=evidence['checkpoint_sha256'],
                      transitions=warm['length']+reference['length']+len(child['trace']['actions'])+evaluation['transitions'],
                      frozen_evaluation=evaluation,
                      max_abs_differences=differences,restored_final_counters=expected,
                      limitation='Tested episode-boundary continuation only; no mid-episode simulator restore or universal GPU determinism claim.')
    except Exception as e: report.update(error=repr(e),traceback=traceback.format_exc())
    report['wall_seconds']=time.monotonic()-start
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
    return 0 if report['status']=='PASS_COLLECTOR_BOUNDARY' else 1

if __name__=='__main__': sys.exit(main())
