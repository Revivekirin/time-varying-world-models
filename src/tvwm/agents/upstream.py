"""Load official unmodified TD-MPC2 in an isolated runner process."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def verify_sources():
    pins = json.loads((ROOT/'environment/sources.lock.json').read_text())
    for name, pin in pins.items():
        repo = ROOT/'external'/name
        sha = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True)
        if sha != pin['commit'] or dirty:
            raise RuntimeError(f'{name} source drift; expected clean {pin["commit"]}')
    return pins


def load_api():
    verify_sources()
    root = ROOT/'external/tdmpc2/tdmpc2'
    if 'common' in sys.modules or 'tdmpc2' in sys.modules:
        raise RuntimeError('use a fresh process; upstream imports are not namespaced')
    sys.path.insert(0, str(root))
    return importlib.import_module('tdmpc2').TDMPC2, importlib.import_module('common.buffer').Buffer


def config(task, obs_dim, action_dim, episode_length):
    from omegaconf import OmegaConf
    from common import MODEL_SIZE
    from common.parser import cfg_to_dataclass
    cfg = OmegaConf.load(ROOT/'external/tdmpc2/tdmpc2/config.yaml')
    overrides = json.loads((ROOT/'configs/algorithm/tdmpc2_smoke.json').read_text())
    for key, value in MODEL_SIZE[overrides['model_size']].items():
        cfg[key] = value
    for key, value in overrides.items():
        cfg[key] = value
    cfg.task = task
    cfg.tasks = [task]
    cfg.task_dim = 0
    cfg.multitask = False
    cfg.obs_shape = {'state': [obs_dim]}
    cfg.action_dim = action_dim
    cfg.episode_length = episode_length
    cfg.bin_size = (cfg.vmax-cfg.vmin)/(cfg.num_bins-1)
    # Runner-only missing upstream fields are resolved explicitly, never passed to upstream runners.
    for key in list(cfg.keys()):
        if OmegaConf.is_missing(cfg, key):
            cfg[key] = None
    return cfg_to_dataclass(cfg)


def episode_tensor(observations, actions, rewards, terminated, truncated):
    import torch
    from tensordict import TensorDict
    from tvwm.envs.contracts import sequence_rows
    sequence_rows(observations, actions, rewards, terminated, truncated)
    n = len(actions)
    return TensorDict(dict(obs=torch.stack(observations).cpu(),
        action=torch.stack([torch.full_like(actions[0], float('nan'))]+actions).cpu(),
        reward=torch.tensor([float('nan')]+[float(r) for r in rewards]),
        terminated=torch.tensor([float('nan')]+[float(t) for t in terminated])), batch_size=[n+1])
