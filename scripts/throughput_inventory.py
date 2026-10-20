"""Read-only source/metric evidence collection. Never imports CUDA or starts training."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True); p.add_argument('--drl',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    result=dict(classification='VERIFIED file contents; no execution or profiler inference')
    for name in ('latest.json','progress.json','wall-ledger.json','provenance.json','resolved_config.json'):
        path=a.run/name
        if path.exists(): result[name]=json.loads(path.read_text())
    latest=a.run/'latest.pt'
    if latest.exists():
        before=(a.run/'latest.json').read_bytes(); h=digest(latest); after=(a.run/'latest.json').read_bytes()
        result['checkpoint_snapshot']=dict(sha256=h,
            stable_sidecar=before==after,hash_matches=before==after and h==json.loads(after)['sha256'],
            note='Read-only live observation; re-audit an immutable checkpoint before migration')
    ledger=result.get('wall-ledger.json')
    if ledger:
        result['charged_wall_seconds_observed']=ledger['charged_seconds']+(max(0,time.time()-ledger['started_epoch']) if ledger['active'] else 0)
    attempts=[]
    for path in sorted(a.run.glob('metrics-*.jsonl')):
        counts=dict(episodes=0,transitions=0,updates=0,initialization_seconds=0.,shutdown_seconds=0.,planner_seconds=0.,learner_seconds=0.)
        first=None; last=None; truncated=0
        for line in path.open():
            try: row=json.loads(line)
            except json.JSONDecodeError: truncated+=1; continue
            if row.get('kind')=='episode':
                counts['episodes']+=1; counts['transitions']+=row['length']
                counts['initialization_seconds']+=row.get('environment_recreation_seconds',0.)
                counts['shutdown_seconds']+=row.get('simulator_worker',{}).get('shutdown_seconds',0.)
                counts['planner_seconds']+=row.get('planner_seconds',0.)
                first=row if first is None else first; last=row
            elif row.get('kind')=='update':
                counts['updates']+=1; counts['learner_seconds']+=row['seconds']
        attempts.append(dict(file=str(path),**counts,first_completed_episode=first,last_completed_episode=last,
            partial_json_lines=truncated,
            learner_updates_per_timed_update_second=counts['updates']/counts['learner_seconds'] if counts['learner_seconds'] else None,
            limitation='No separate physics/IPC/checkpoint timers. Do not label sum of timed components as true throughput.'))
    result['attempts']=attempts
    if a.drl.is_dir():
        result['drl_commit']=subprocess.run(['git','-C',str(a.drl),'rev-parse','HEAD'],text=True,capture_output=True).stdout.strip()
        listing=subprocess.run(['git','-C',str(a.drl),'ls-files'],text=True,capture_output=True,check=True).stdout.splitlines()
        pattern=re.compile(r'ManiSkill|mani_skill|num_envs|autoreset|auto_reset|final_observation|termination|truncat|replay|update.to.data|utd|td.?mpc|sim_backend|\.cpu\(|\.cuda\(|device|control_freq|max_episode_steps',re.I)
        matches=[]; selected=set()
        for relative in listing:
            path=a.drl/relative
            if path.is_symlink() or path.suffix not in ('.py','.yaml','.yml','.toml','.txt') or not path.is_file() or path.stat().st_size>512000: continue
            lines=path.read_text(errors='replace').splitlines()
            hits=[i for i,line in enumerate(lines) if pattern.search(line)]
            if not hits: continue
            indices=set(j for i in hits for j in range(max(0,i-3),min(len(lines),i+4)))
            selected.add(relative)
            matches.append(dict(path=relative,sha256=digest(path),lines=[f'{i+1}: {lines[i]}' for i in sorted(indices)]))
        result['drl_scan']=dict(scanned_tracked_files=len(listing),matched_files=sorted(selected),matches=matches,
            limitation='Search excerpts need interpretation; absence of a name is not proof an implementation is absent.')
    else: result['drl_scan']=dict(status='UNRESOLVED',reason='path unavailable')
    (a.out/'inventory.json').write_text(json.dumps(result,indent=2)+'\n')
    print(a.out/'inventory.json')

if __name__=='__main__': main()
