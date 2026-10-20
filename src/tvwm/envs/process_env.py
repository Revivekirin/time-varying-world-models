"""Episode-local spawned simulator. No CUDA tensors cross the process boundary."""
import copy
import os
import signal
import sys
import multiprocessing as mp
import time
import queue
import threading
import traceback
from tvwm.envs.contracts import compare_recorded_contract


class WorkerError(RuntimeError):
    pass


class WorkerTransport:
    """One request in flight; bounded receive/shutdown and explicit native-exit errors."""
    def __init__(self, target, args=(), timeout=120.):
        self.timeout=timeout
        context=mp.get_context('spawn')
        self.connection, child=context.Pipe()
        self.process=context.Process(target=target,args=(child,*args),daemon=True)
        try:
            self.process.start()
        except BaseException:
            self.connection.close()
            raise
        finally:
            child.close()

    def receive(self):
        # poll() alone does not bound recv() if a child dies/hangs mid-frame.
        result=queue.Queue(maxsize=1)
        def read():
            try: result.put((True,self.connection.recv()))
            except BaseException as error: result.put((False,error))
        reader=threading.Thread(target=read,daemon=True)
        reader.start()
        try:
            try: ok,message=result.get(timeout=self.timeout)
            except queue.Empty:
                raise WorkerError(f'worker timeout after {self.timeout}s; pid={self.process.pid}, exit={self.process.exitcode}')
            if not ok: raise message
            if not isinstance(message,dict) or message.get('status') not in ('ok','error'):
                raise WorkerError('invalid worker IPC response')
            if message['status']=='error':
                raise WorkerError(message['error']+'\n'+message.get('traceback',''))
            return message['payload']
        except BaseException as error:
            self.abort()
            if isinstance(error,(EOFError,OSError)):
                raise WorkerError(f'worker IPC/native exit; exit={self.process.exitcode}: {error}') from error
            raise
        finally:
            reader.join(5)

    def request(self, command, payload=None):
        try:
            self.connection.send((command,payload))
        except (OSError,EOFError) as error:
            self.abort()
            raise WorkerError(f'worker send failed; exit={self.process.exitcode}: {error}') from error
        return self.receive()

    def abort(self):
        self.connection.close()
        if self.process.is_alive(): self.process.terminate()
        self.process.join(5)
        if self.process.is_alive():
            self.process.kill(); self.process.join(5)
        if self.process.is_alive(): raise WorkerError('worker could not be reaped after kill')

    def close(self):
        try:
            result=self.request('close')
            self.process.join(10)
            if self.process.is_alive() or self.process.exitcode!=0:
                raise WorkerError(f'worker shutdown failed; exit={self.process.exitcode}')
            return result
        finally:
            self.abort()


def simulator_worker(connection, task, seed, parent_pid):
    env=None
    try:
        # A killed/native-crashed learner must not leave a CUDA simulator orphan.
        if sys.platform=='linux':
            import ctypes
            if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)!=0:
                raise OSError(ctypes.get_errno(),'cannot set simulator parent-death signal')
            if os.getppid()!=parent_pid: return
        import torch
        from tvwm.agents.nominal import seed_all, goal_distance
        from tvwm.agents.resume import rng_state, restore_rng
        from tvwm.envs.maniskill import ManiSkillAdapter
        # Match the previous constructor/reset RNG ordering inside this process.
        seed_all(seed)
        state=rng_state()
        env=ManiSkillAdapter(task)
        restore_rng(state)
        obs,info=env.reset(seed=seed)

        def memory():
            import subprocess
            free,total=torch.cuda.mem_get_info()
            rows=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free',
                '--format=csv,noheader,nounits'],text=True,timeout=5).strip().splitlines()
            if len(rows)!=1: raise WorkerError('single GPU required by existing resource policy')
            used,available=map(int,rows[0].split(','))
            if used>20*1024 or min(free,available*1024**2)<4*1024**3:
                raise WorkerError('simulator worker GPU memory budget exceeded')
            return dict(cuda_free_bytes=free,cuda_total_bytes=total,device_used_mib=used,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())

        def send(payload): connection.send(dict(status='ok',payload=payload))
        def observation(value):
            if hasattr(value,'detach'): value=value.detach().cpu().numpy()
            return value.copy()
        send(dict(obs=observation(obs),contract=env.contract,goal_distance=goal_distance(env),memory=memory()))
        while True:
            command,payload=connection.recv()
            if command=='step':
                obs,reward,term,trunc,info=env.step(payload)
                if torch.cuda.mem_get_info()[0]<4*1024**3: raise WorkerError('worker free CUDA memory below 4 GiB')
                send(dict(obs=observation(obs),reward=float(reward),terminated=term,truncated=trunc,
                          success=bool(info['success'].item()),goal_distance=goal_distance(env)))
            elif command=='render':
                state=rng_state()
                try: frame=observation(env.base.render_rgb_array())
                finally: restore_rng(state)
                send(frame)
            elif command=='close':
                stats=memory()
                env.close(); env=None
                torch.cuda.synchronize()
                send(stats)
                break
            else: raise WorkerError(f'unknown simulator command: {command}')
    except BaseException as error:
        try: connection.send(dict(status='error',error=repr(error),traceback=traceback.format_exc()))
        except (OSError,EOFError): pass
    finally:
        if env is not None:
            try: env.close()
            except BaseException: pass
        connection.close()


class EpisodeEnvironment:
    lifecycle='spawn_per_episode_v1'

    def __init__(self, task):
        self.task=task; self._worker=None; self._contract=None
        self.done=True; self._has_stepped=False
        self.last_recreation_seconds=0.; self.last_reset_seconds=0.; self.recreation_count=0
        self.worker_metrics={}

    def reset(self, seed):
        import torch
        from tvwm.agents.resume import rng_state, restore_rng
        if self._has_stepped and not self.done: raise RuntimeError('cannot recreate midway through an episode')
        started=time.monotonic(); state=rng_state()
        try:
            self.close()
            self._worker=WorkerTransport(simulator_worker,(self.task,seed,os.getpid()))
            data=self._worker.receive()
            if self._contract is not None:
                compare_recorded_contract(data['contract'],self._contract,list(self._contract))
            self._contract=copy.deepcopy(data['contract']); self.contract=data['contract']
            self.action_dim=self.contract['action_dim']; self.horizon=self.contract['horizon']
            self._distance=data['goal_distance']; self.done=False; self._has_stepped=False
            self.worker_metrics=dict(pid=self._worker.process.pid,startup=data['memory'])
        except BaseException:
            if self._worker is not None: self._worker.abort(); self._worker=None
            raise
        finally: restore_rng(state)
        self.last_recreation_seconds=self.last_reset_seconds=time.monotonic()-started
        self.recreation_count+=1
        return torch.from_numpy(data['obs']).to('cuda'),{}

    def step(self, action):
        import torch
        if self._worker is None or self.done: raise RuntimeError('reset required before step')
        try:
            data=self._worker.request('step',action.detach().cpu().tolist())
        except BaseException:
            self._worker.abort(); self._worker=None
            raise
        self.done=data['terminated'] or data['truncated']; self._has_stepped=True
        self._distance=data['goal_distance']
        return (torch.from_numpy(data['obs']).to('cuda'),data['reward'],data['terminated'],data['truncated'],
                {'success':torch.tensor(data['success'])})

    def goal_distance(self): return self._distance

    def render_rgb_array(self):
        try: return self._worker.request('render')
        except BaseException:
            self._worker.abort(); self._worker=None
            raise

    def finish_episode(self):
        if not self.done: raise RuntimeError('cannot commit an incomplete simulator episode')
        self.close()

    def close(self):
        worker,self._worker=self._worker,None
        if worker is not None:
            started=time.monotonic()
            self.worker_metrics['shutdown']=worker.close()
            self.worker_metrics['shutdown_seconds']=time.monotonic()-started
