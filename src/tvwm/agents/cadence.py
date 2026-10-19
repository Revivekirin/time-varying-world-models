"""Pinned upstream zero-based collection/update cadence, with an exact transition cap."""
def updates_at(step, seed_steps=1000):
    if step < 0 or seed_steps < 1:
        raise ValueError('invalid step or seed_steps')
    if step < seed_steps:
        return 0
    return seed_steps if step == seed_steps else 1


def random_action_at(step, seed_steps=1000):
    if step < 0 or seed_steps < 1:
        raise ValueError('invalid step or seed_steps')
    return step <= seed_steps
