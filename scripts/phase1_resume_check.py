"""Fresh-process learner resume audit: zero simulation transitions, one batch256 update."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''): h.update(data)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--preflight',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',transitions=0,updates_executed=0,full_training_authorized=False)
    start=time.monotonic()
    def timeout(*_):raise TimeoutError('fresh-process audit 180s timeout')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(180)
    try:
        import torch
        import importlib.metadata as md
        from tvwm.agents.upstream import load_api,verify_sources
        from tvwm.agents.resume import load_learner
        evidence=json.loads((a.preflight/'report.json').read_text())
        if evidence['status']!='PASS_COMPATIBILITY_ONLY': raise ValueError('successful preflight required')
        report['task']=evidence['task']
        pins=verify_sources()
        if pins!=evidence['sources']: raise ValueError('source mismatch')
        packages={k:md.version(k) for k in evidence['packages']}
        if packages!=evidence['packages']:raise ValueError('package mismatch')
        checkpoint=a.preflight/'learner-envelope.pt'
        if sha256(checkpoint)!=evidence['checkpoint_sha256']:raise ValueError('checkpoint hash mismatch')
        if sha256(ROOT/'environment/requirements-linux.lock')!=evidence['dependency_lock_sha256']:
            raise ValueError('dependency lock mismatch')
        Agent,Buffer=load_api()
        from omegaconf import OmegaConf
        from common.parser import cfg_to_dataclass
        from torchrl.data.replay_buffers import LazyTensorStorage
        values=json.loads((a.preflight/'resolved_config.json').read_text())
        if values['batch_size']!=256 or values['model_size']!=5:raise ValueError('unexpected model/batch')
        cfg=cfg_to_dataclass(OmegaConf.create(values))
        agent=Agent(cfg)
        replay=Buffer(cfg)
        # Fresh storage; unlike prior test no pre-existing replay contents are available.
        replay._storage_device=torch.device(evidence['replay_storage_device'])
        replay._buffer=replay._reserve_buffer(LazyTensorStorage(replay.capacity,device=replay._storage_device))
        provenance=dict(sources=pins,config=vars(cfg),task=evidence['task'],
                        dependency_lock_sha256=evidence['dependency_lock_sha256'],packages=packages)
        counters=load_learner(checkpoint,agent,replay,provenance)
        expected_counters=dict(transitions=evidence['transitions'],updates=1,episode_index=2)
        if counters!=expected_counters:raise AssertionError('restored counter mismatch')
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.monotonic()
        metrics=agent.update(replay)
        torch.cuda.synchronize();report['updates_executed']=1
        report['update_seconds']=time.monotonic()-t
        metrics={k:float(v) for k,v in metrics.items()}
        reference=evidence['losses'][1]
        if metrics.keys()!=reference.keys():raise AssertionError('metric keys mismatch')
        for key,value in metrics.items():
            if not math.isfinite(value) or not math.isclose(value,reference[key],rel_tol=1e-5,abs_tol=1e-6):
                raise AssertionError(f'fresh-process metric differs: {key}: {value} vs {reference[key]}')
        report.update(status='PASS_LEARNER_RESUME_ONLY',losses=metrics,restored_counters=counters,
                      checkpoint_sha256=evidence['checkpoint_sha256'],
                      torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      limitation='No simulator/collector state continuation; full-run launch remains gated.')
    except Exception as e:
        report.update(status='FAIL',error=repr(e),traceback=traceback.format_exc())
    finally:
        report['wall_seconds']=time.monotonic()-start
        (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');signal.alarm(0)
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='PASS_LEARNER_RESUME_ONLY' else 1

if __name__=='__main__':sys.exit(main())
