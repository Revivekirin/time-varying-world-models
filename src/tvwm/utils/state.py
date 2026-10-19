"""Snapshot model state including TensorDict's non-tensor metadata."""
from copy import deepcopy


def snapshot_state(state):
    # deepcopy preserves OrderedDict metadata and tensor alias relationships.
    return deepcopy(state)


def assert_state_equal(actual, expected):
    import torch
    if actual.keys() != expected.keys():
        raise AssertionError('state keys differ')
    for key, value in actual.items():
        other = expected[key]
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, other, rtol=0, atol=0, msg=lambda msg: f'{key}: {msg}')
        elif type(value) is not type(other) or value != other:
            raise AssertionError(f'state metadata differs at {key}: {value!r} != {other!r}')
