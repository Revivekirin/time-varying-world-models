"""Opt-in candidate: overlap CPU imports only, never two live GPU simulators."""
import copy
import os
import signal
import sys
import time
import traceback
from tvwm.envs.process_env import EpisodeEnvironment as BaselineEnvironment, WorkerTransport, simulator_worker
from tvwm.envs.contracts import compare_recorded_contract


def cpu_imports():
    started=time.monotonic()
    import torch
    import resource
    from tvwm.envs.maniskill import ManiSkillAdapter
    from tvwm.agents import nominal, resume
    if torch.cuda.is_initialized(): raise RuntimeError('CPU preparation unexpectedly initialized CUDA')
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(cpu_import_seconds=time.monotonic()-started,
                peak_rss_bytes=rss*(1 if sys.platform=='darwin' else 1024))


def prepared_worker(connection, parent_pid, load=cpu_imports, activate=simulator_worker):
    try:
        if sys.platform=='linux':
            import ctypes
            if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)!=0:
                raise OSError(ctypes.get_errno(),'cannot set parent-death signal')
            if os.getppid()!=parent_pid: return
        connection.send(dict(status='ok',payload=load()))
        command,payload=connection.recv()
        if command!='activate': raise RuntimeError('activation required after CPU preparation')
        task,seed=payload
        activate(connection,task,seed,parent_pid)
    except BaseException as error:
        try: connection.send(dict(status='error',error=repr(error),traceback=traceback.format_exc()))
        except (OSError,EOFError): pass
    finally: connection.close()


class PrefetchedEnvironment(BaselineEnvironment):
    lifecycle='spawn_cpu_prefetch_v1'

    def __init__(self,task):
        super().__init__(task)
        self._prepared=None

    def reset(self,seed):
        import torch
        from tvwm.agents.resume import rng_state,restore_rng
        if self._has_stepped and not self.done: raise RuntimeError('cannot recreate midway through an episode')
        started=time.monotonic(); state=rng_state()
        try:
            # Close only the old active simulator; preserve prepared CPU process.
            BaselineEnvironment.close(self)
            self._worker,self._prepared=self._prepared,None
            if self._worker is None: self._worker=WorkerTransport(prepared_worker,(os.getpid(),))
            preparation=self._worker.receive()
            activated=time.monotonic()
            data=self._worker.request('activate',(self.task,seed))
            if self._contract is not None:
                compare_recorded_contract(data['contract'],self._contract,list(self._contract))
            self._contract=copy.deepcopy(data['contract']); self.contract=data['contract']
            self.action_dim=self.contract['action_dim']; self.horizon=self.contract['horizon']
            self._distance=data['goal_distance']; self.done=False; self._has_stepped=False
            self.worker_metrics=dict(pid=self._worker.process.pid,startup=data['memory'],
                cpu_preparation=preparation,activation_seconds=time.monotonic()-activated)
            # This process imports Python modules only. Its CUDA context is created
            # at the NEXT reset, after the previous GPU worker has exited.
            self._prepared=WorkerTransport(prepared_worker,(os.getpid(),))
        except BaseException:
            if self._worker is not None: self._worker.abort(); self._worker=None
            if self._prepared is not None: self._prepared.abort(); self._prepared=None
            raise
        finally: restore_rng(state)
        self.last_recreation_seconds=self.last_reset_seconds=time.monotonic()-started
        self.recreation_count+=1
        return torch.from_numpy(data['obs']).to('cuda'),{}

    def finish_episode(self):
        if not self.done: raise RuntimeError('cannot commit an incomplete simulator episode')
        BaselineEnvironment.close(self)

    def close(self):
        try: BaselineEnvironment.close(self)
        finally:
            prepared,self._prepared=self._prepared,None
            if prepared is not None: prepared.abort()


def environment_class(pipeline):
    if pipeline=='baseline': return BaselineEnvironment
    if pipeline=='cpu-prefetch': return PrefetchedEnvironment
    raise ValueError(f'unknown simulator pipeline: {pipeline}')
