"""Common local/W&B metric schema. One W&B run per execution attempt prevents rollback overlap."""
import json
from pathlib import Path


def scalar_metrics(kind, data, counters):
    axes={'step/env':counters['transitions'],'step/update':counters['updates'],
          'step/eval_env':counters.get('evaluation_transitions',0)}
    keys={
        'episode':{'return_':'train/episode_return','success':'train/success','length':'train/episode_length',
                   'goal_progress':'train/goal_progress','final_goal_distance':'train/final_goal_distance',
                   'environment_recreation_seconds':'time/environment_recreation_s','wall_seconds':'time/wall_s'},
        'validation':{'success_rate':'eval/success_rate','mean_return':'eval/return_mean',
                      'return_std':'eval/return_std','goal_progress_mean':'eval/goal_progress_mean',
                      'terminal_success_rate':'eval/terminal_success_rate'},
        'resources':{'device_used_mib':'gpu/device_used_mib','device_free_mib':'gpu/device_free_mib',
                     'peak_allocated_bytes':'gpu/peak_allocated_bytes','peak_reserved_bytes':'gpu/peak_reserved_bytes',
                     'storage_free_bytes':'system/storage_free_bytes'}}
    if kind=='update':
        axes.update({f'loss/{k}':float(v) for k,v in data.items() if k not in ('transitions','updates','seconds')})
        axes['perf/learner_updates_per_s']=1/max(data['seconds'],1e-12)
    else:
        axes.update({dst:float(data[src]) for src,dst in keys.get(kind,{}).items() if src in data})
    if kind=='episode':
        axes['perf/planner_ms_per_transition']=1000*data['planner_seconds']/max(data['length'],1)
        worker=data.get('simulator_worker',{})
        if 'shutdown_seconds' in worker: axes['time/worker_shutdown_s']=worker['shutdown_seconds']
        for stage in ('startup','shutdown'):
            for key,value in worker.get(stage,{}).items(): axes[f'gpu/worker_{stage}_{key}']=value
    return axes


class Monitor:
    def __init__(self,out,task,attempt,counters,provenance,mode,project,entity=None):
        from tqdm.auto import tqdm
        self.out=Path(out);self.counters=counters;self.run=None;self.latest={}
        self.events=(self.out/f'tracking-{attempt}.jsonl').open('a',buffering=1)
        if mode!='disabled':
            from tvwm.agents.resume import rng_state,restore_rng
            state=rng_state()
            try:
                import wandb
                self.run=wandb.init(project=project,entity=entity,mode=mode,dir=str(out),
                    name=f'{task}-seed1-{attempt}',group=f'{task}-seed1',job_type='nominal-training',
                    config=dict(schema_version=1,attempt=attempt,restored_counters=dict(counters),provenance=provenance))
                self.run.define_metric('step/env');self.run.define_metric('step/update')
                for pattern in ('train/*','eval/*','gpu/*','time/*','system/*','perf/*','checkpoint/*'):
                    self.run.define_metric(pattern,step_metric='step/env')
                self.run.define_metric('loss/*',step_metric='step/update')
                (self.out/f'wandb-{attempt}.json').write_text(json.dumps(dict(id=self.run.id,url=self.run.url,mode=mode,group=f'{task}-seed1'),indent=2))
            finally:restore_rng(state)
        self.bar=tqdm(total=300000,initial=counters['transitions'],desc=task,unit='trans',dynamic_ncols=True,mininterval=1)
        self.updates=tqdm(total=299999,initial=counters['updates'],desc='Learner',unit='upd',dynamic_ncols=True,mininterval=1,position=1)

    def phase(self,label):
        self.bar.set_description(label);self.bar.refresh()

    def step(self,counters):
        self.bar.update(counters['transitions']-self.bar.n)

    def learner(self,data):
        self.updates.update(data['updates']-self.updates.n)
        self.updates.set_postfix(loss=f"{data.get('total_loss',0):.4f}",refresh=False)

    def emit(self,kind,data):
        record=scalar_metrics(kind,data,self.counters)
        self.events.write(json.dumps(dict(kind=kind,**record),allow_nan=False)+'\n')
        if kind=='episode':
            self.latest.update(return_=f"{data['return_']:.2f}",success=int(data['success']))
        if kind=='validation':
            self.latest.update(val_success=f"{data['success_rate']:.1%}",val_return=f"{data['mean_return']:.2f}")
        self.bar.set_postfix(self.latest,refresh=False)
        if self.run:
            from tvwm.agents.resume import rng_state,restore_rng
            state=rng_state()
            try:
                if kind=='validation':
                    import wandb
                    for i,video in enumerate(data.get('videos',[])):
                        record[f'eval/video_{i}']=wandb.Video(video['path'],format='mp4',caption=f"seed={video['seed']} success={video['success']} return={video['return_']:.3f}")
                    columns=['seed','return_','success','terminal_success','length','goal_progress','final_goal_distance']
                    record['eval/episodes']=wandb.Table(columns=columns,data=[[ep[k] for k in columns] for ep in data['episodes']])
                self.run.log(record)
            finally:restore_rng(state)

    def checkpoint(self,name,sha):
        self.events.write(json.dumps(dict(kind='checkpoint',name=name,sha256=sha,**scalar_metrics('checkpoint',{},self.counters)))+'\n')
        if self.run:
            from tvwm.agents.resume import rng_state,restore_rng
            state=rng_state()
            try:self.run.summary['checkpoint/'+name]=sha
            finally:restore_rng(state)

    def close(self,status):
        self.bar.close();self.updates.close();self.events.close()
        if self.run:
            self.run.summary['status']=status
            self.run.finish(exit_code=1 if status=='FAIL' else 0)


class EvaluationVideo:
    """Stream one existing evaluation episode to disk; no extra rollouts."""
    def __init__(self,path,fps=20):
        from tvwm.agents.resume import rng_state,restore_rng
        state=rng_state()
        try:
            import imageio.v2 as imageio
            self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
            self.writer=imageio.get_writer(str(path),fps=fps,codec='libx264',macro_block_size=1)
        finally:restore_rng(state)

    def frame(self,env):
        import numpy as np
        from tvwm.agents.resume import rng_state,restore_rng
        state=rng_state()
        try:
            pixels=env.render_rgb_array() if hasattr(env,'render_rgb_array') else env.base.render_rgb_array()
            if pixels is None: raise RuntimeError('evaluation renderer returned no RGB frame')
            if hasattr(pixels,'detach'):pixels=pixels.detach().cpu().numpy()
            pixels=np.asarray(pixels)
            if pixels.ndim==4 and pixels.shape[0]==1:pixels=pixels[0]
            if pixels.ndim!=3 or pixels.shape[-1]!=3 or pixels.dtype!=np.uint8:
                raise ValueError(f'expected uint8 HWC RGB video frame; got {pixels.shape}/{pixels.dtype}')
            self.writer.append_data(pixels)
        finally:restore_rng(state)

    def close(self):self.writer.close()
