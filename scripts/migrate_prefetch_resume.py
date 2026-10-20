"""Audited 80,011-step baseline-to-prefetch copy; never launches training."""
import argparse
import importlib.metadata as md
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from phase1_resume_check import sha256

EXPECTED_CHECKPOINT='b3c137fe29cf5d46594ad086ac671b8ca337946c31321b815a88fadb495e5a75'
# Original checkpoint implementation; only this reviewed transition is accepted.
EXPECTED_CANDIDATE_CODE='c473111a653892bbfbcbb979166861e7e2d1e5250d202e6afae33862938761c4'
EXPECTED_OLD_CODE='602960be96796089a494f0a7606bb88483323d9bfb6c8f68ad9072e7ded31c7c'


def exact(actual, expected, path='state'):
    import numpy as np
    import torch
    if isinstance(expected,torch.Tensor):
        if not isinstance(actual,torch.Tensor) or actual.dtype!=expected.dtype:
            raise AssertionError(f'{path}: tensor type/dtype mismatch')
        torch.testing.assert_close(actual.cpu(),expected.cpu(),rtol=0,atol=0,equal_nan=True)
    elif isinstance(expected,np.ndarray):
        if actual.dtype!=expected.dtype: raise AssertionError(f'{path}: numpy dtype')
        np.testing.assert_array_equal(actual,expected)
    elif isinstance(expected,dict):
        if actual.keys()!=expected.keys(): raise AssertionError(f'{path}: keys')
        for k in expected: exact(actual[k],expected[k],f'{path}/{k}')
    elif isinstance(expected,(list,tuple)):
        if type(actual)!=type(expected) or len(actual)!=len(expected): raise AssertionError(f'{path}: sequence')
        for i,(a,b) in enumerate(zip(actual,expected)): exact(a,b,f'{path}/{i}')
    elif type(actual)!=type(expected) or actual!=expected:
        raise AssertionError(f'{path}: metadata mismatch')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--push-gate',type=Path,required=True); p.add_argument('--pick-gate',type=Path,required=True)
    p.add_argument('--performance',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    report=dict(status='NO_GO',training_launched=False,optimizer_updates=0,simulation_transitions=0)
    read=lambda path:json.loads(path.read_text())
    try:
        import torch
        from tvwm.agents.nominal import code_hash,empty_replay,atomic_json
        from tvwm.agents.upstream import load_api,verify_sources
        from tvwm.agents.resume import load_learner,save_learner
        from phase1_train import resources
        plan=read(ROOT/'configs/phase1/nominal.json')
        import subprocess
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if active: raise ValueError('GPU processes active; migration requires an idle GPU')
        if code_hash()!=EXPECTED_CANDIDATE_CODE: raise ValueError('candidate implementation differs from reviewed collector evidence')
        performance=read(a.performance)
        if (performance.get('status')!='PROMISING_FROZEN_COLLECTION_ONLY'
            or performance.get('strict_traces_and_rng')!='PASS'
            or performance.get('relative_gain',0)<.10
            or performance.get('checkpoint_sha256')!=EXPECTED_CHECKPOINT
            or performance.get('code_sha256')!=code_hash()):
            raise ValueError('matching Push frozen-collection performance evidence required')
        report['performance_evidence_sha256']=sha256(a.performance)
        report['performance_scope']='frozen collection only; full learner throughput remains UNRESOLVED'
        resources(a.out,plan)
        for gate,task in [(a.push_gate,'PushCube-v1'),(a.pick_gate,'PickCube-v1')]:
            g=read(gate/'report.json')
            if (g['status'],g['task'],g['environment_lifecycle'],g['code_sha256'])!=(
                'PASS_COLLECTOR_BOUNDARY',task,'spawn_cpu_prefetch_v1',code_hash()):
                raise ValueError('both current prefetch collector gates required')
        source=a.source/'latest.pt'; metadata=read(a.source/'latest.json')
        digest=sha256(source); report['original_checkpoint_sha256']=digest
        if digest!=EXPECTED_CHECKPOINT or digest!=metadata['sha256']:
            raise ValueError('not the explicitly identified 80,011-transition checkpoint')
        ledger=read(a.source/'wall-ledger.json'); progress=read(a.source/'progress.json')
        charged=max(metadata['counters']['wall_seconds'],progress['wall_seconds'],
                    ledger['charged_seconds']+(max(0.,time.time()-ledger['started_epoch']) if ledger['active'] else 0.))
        report['charged_wall_seconds_at_audit']=charged
        gate=read(a.push_gate/'report.json')
        startup=gate['recreation_overhead_estimate']['mean_measured_seconds']
        c=metadata['counters']; remaining=300000-c['transitions']
        report['runtime_projection']=dict(classification='INFERRED, conservative; future episode lengths may change',
            estimated_remaining_episodes=remaining*c['episode_index']/c['transitions'],
            estimated_remaining_seconds=remaining/c['transitions']*(c['wall_seconds']+startup*c['episode_index']),
            available_seconds=max(0.,86400-charged),startup_seconds_per_episode=startup)
        if charged>=plan['max_wall_seconds_per_task']-900:
            raise ValueError('existing cumulative 24-hour budget exhausted; downtime accounting preserved')
        saved=torch.load(source,map_location='cpu',weights_only=False)
        old=saved['provenance']; report['original_code_sha256']=old['code_sha256']
        if saved['format_version']!=1 or old['code_sha256']!=EXPECTED_OLD_CODE:
            raise ValueError('original implementation provenance unverified; no permissive migration')
        if old['environment_lifecycle']!='spawn_per_episode_v1' or old['task']!='PushCube-v1':
            raise ValueError('unexpected original lifecycle/task')
        if old!=read(a.source/'provenance.json'): raise ValueError('provenance sidecar mismatch')
        if old['sources']!=verify_sources() or old['plan']!=plan: raise ValueError('source/plan drift')
        if old['packages']!={k:md.version(k) for k in old['packages']}: raise ValueError('package drift')
        if old['dependency_lock_sha256']!=sha256(ROOT/'environment/requirements-linux.lock'): raise ValueError('lock drift')
        exact(saved['counters'],metadata['counters'])
        counters=saved['counters']
        if (counters['transitions'],counters['updates'],counters['episode_index'],counters['boundary'])!=(80011,80010,2827,True):
            raise ValueError('unexpected checkpoint counters/boundary')
        if old['config']!=read(a.source/'resolved_config.json'): raise ValueError('resolved config drift')
        new=dict(old,environment_lifecycle='spawn_cpu_prefetch_v1',code_sha256=code_hash())
        Agent,Buffer=load_api()
        from omegaconf import OmegaConf
        from common.parser import cfg_to_dataclass
        cfg=cfg_to_dataclass(OmegaConf.create(old['config']))
        if vars(cfg)!=old['config']: raise ValueError('resolved configuration not round-trippable')
        agent=Agent(cfg); replay=empty_replay(Buffer,cfg,metadata['replay_device'])
        restored=load_learner(source,agent,replay,old)
        save_learner(a.out/'restoration-audit.pt',agent,replay,restored,old)
        roundtrip=torch.load(a.out/'restoration-audit.pt',map_location='cpu',weights_only=False)
        exact(roundtrip,saved)
        report['state_restoration']='VERIFIED exact model, both optimizers, scale, planner, replay, RNG and counters'
        # Only provenance changes. Never rewrite original artifacts or training counters.
        save_learner(a.out/'latest.pt',agent,replay,restored,new)
        converted=torch.load(a.out/'latest.pt',map_location='cpu',weights_only=False)
        expected=dict(saved,provenance=new); exact(converted,expected)
        if sha256(source)!=digest: raise ValueError('original checkpoint changed during audit')
        selected=a.source/counters['best_model_file']
        if selected.parent!=a.source or sha256(selected)!=counters['best_model_sha256']:
            raise ValueError('best model path/hash mismatch')
        names=[counters['best_model_file'],counters['best_validation_file'],counters['best_envelope_file']]
        if counters['best_envelope_file']:
            ep=a.source/counters['best_envelope_file']; side=ep.with_suffix('.json')
            if sha256(ep)!=read(side)['sha256']: raise ValueError('best envelope hash mismatch')
            names.append(side.name)
        for name in names+['wall-ledger.json','progress.json','resolved_config.json','nominal_config.json']:
            if name is None: continue
            if Path(name).name!=name: raise ValueError('unsafe artifact path')
            shutil.copy2(a.source/name,a.out/name)
        atomic_json(a.out/'latest.json',dict(metadata,sha256=sha256(a.out/'latest.pt')))
        atomic_json(a.out/'provenance.json',new)
        report['training_authorized']=False
        report['remaining_blocker']='Current candidate runner blocks full prefetch training. This audit does not remove that guard.'
        report.update(status='PASS_RESUME_PREPARED',new_checkpoint_sha256=sha256(a.out/'latest.pt'),
                      changed_provenance_fields=['environment_lifecycle','code_sha256'],counters=counters,
                      limitation='No universal determinism claim; existing wall ledger and rollback semantics retained.')
    except Exception as error:
        report.update(error=repr(error),traceback=traceback.format_exc())
    (a.out/'prefetch-migration-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='PASS_RESUME_PREPARED' else 1

if __name__=='__main__': sys.exit(main())
