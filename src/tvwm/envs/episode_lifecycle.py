"""Phase 1 boundary-only simulator recreation; learner state is never owned here."""
from contextlib import contextmanager
import copy
import gc
import time
from tvwm.envs.contracts import compare_recorded_contract


@contextmanager
def construction_rng(seed):
    # Snapshot the CURRENT stream, not the checkpoint's old stream. Construction
    # gets a repeatable seed; restore before the ordinary episode reset runs.
    from tvwm.agents.resume import rng_state, restore_rng
    import random
    import numpy as np
    import torch
    state=rng_state()
    try:
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        yield
    finally:
        restore_rng(state)


def synchronize():
    import torch
    torch.cuda.synchronize()


def adapter_factory(task):
    from tvwm.envs.maniskill import ManiSkillAdapter
    return ManiSkillAdapter(task)


class EpisodeEnvironment:
    """One live adapter at a time, replaced before every new episode reset."""
    lifecycle='recreate_before_episode_v1'

    def __init__(self, task):
        self.task=task
        self._adapter=None
        self._contract=None
        self._has_stepped=False
        self.last_recreation_seconds=0.
        self.last_reset_seconds=0.
        self.recreation_count=0

    def __getattr__(self, name):
        adapter=self.__dict__.get('_adapter')
        if adapter is None: raise AttributeError(f'environment not initialized: {name}')
        return getattr(adapter,name)

    def close(self):
        old=self._adapter
        self._adapter=None
        if old is not None:
            synchronize()
            try:
                old.close()
            finally:
                del old
                gc.collect()
        self._has_stepped=False

    def reset(self, seed):
        if self._adapter is not None and self._has_stepped and not self._adapter.done:
            raise RuntimeError('cannot recreate midway through an episode')
        started=time.monotonic()
        with construction_rng(seed):
            self.close()
            self._adapter=adapter_factory(self.task)
            synchronize()
        self.last_recreation_seconds=time.monotonic()-started
        try:
            result=self._adapter.reset(seed=seed)
            if self._contract is None:
                self._contract=copy.deepcopy(self._adapter.contract)
            else:
                compare_recorded_contract(self._adapter.contract,self._contract,list(self._contract))
        except Exception:
            self.close()
            raise
        self._has_stepped=False
        self.recreation_count+=1
        self.last_reset_seconds=time.monotonic()-started
        return result

    def step(self, action):
        if self._adapter is None: raise RuntimeError('reset required before step')
        result=self._adapter.step(action)
        self._has_stepped=True
        return result
