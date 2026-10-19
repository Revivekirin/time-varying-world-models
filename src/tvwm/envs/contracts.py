"""Dependency-free contract checks; never reinterpret absolute commands as deltas."""
import math

TASKS = ('PushCube-v1', 'PickCube-v1')


def affine_action(values, low, high, *, inverse=False):
    if not (len(values) == len(low) == len(high)):
        raise ValueError('action/bound dimensions differ')
    out = []
    for x, lo, hi in zip(values, low, high):
        if not all(math.isfinite(v) for v in (x, lo, hi)) or lo >= hi:
            raise ValueError('nonfinite action or invalid bounds')
        if inverse:
            if not lo <= x <= hi:
                raise ValueError('native action outside bounds')
            out.append(2 * (x-lo)/(hi-lo)-1)
        else:
            if not -1 <= x <= 1:
                raise ValueError('normalized action outside [-1, 1]')
            out.append(lo + (x+1)*(hi-lo)/2)
    return out


def sequence_rows(observations, actions, rewards, terminated, truncated):
    """Upstream rows: row t+1 stores (o[t+1], a[t], r[t], term[t])."""
    n = len(actions)
    if n < 1 or len(observations) != n+1 or any(len(x) != n for x in (rewards, terminated, truncated)):
        raise ValueError('invalid episode lengths')
    if any(bool(t) or bool(u) for t, u in zip(terminated[:-1], truncated[:-1])):
        raise ValueError('episode boundary inside sequence')
    if not (bool(terminated[-1]) or bool(truncated[-1])):
        raise ValueError('only completed episodes may enter replay')
    return [(observations[0], None, None, None)] + list(zip(observations[1:], actions, rewards, terminated))
