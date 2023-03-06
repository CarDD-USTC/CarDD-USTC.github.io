# Hack MobulaOP for the compatible functions
import torch


def max(data, axis=None, keepdims=None):
    if axis is None:
        return torch.max(data, keepdim=keepdims)[0]
    return torch.max(data, axis, keepdim=keepdims)[0]


def sum(data, axis=None, keepdims=None):
    if axis is None:
        return torch.sum(data, keepdim=keepdims)
    return torch.sum(data, axis, keepdim=keepdims)


def minimum(lhs, rhs, out=None):
    assert isinstance(lhs, torch.Tensor)
    if isinstance(rhs, torch.Tensor):
        return torch.min(lhs, rhs, out=out)
    if out is None:
        return lhs.clamp_max(rhs)
    return lhs.clamp(max=rhs, out=out)


broadcast_minimum = minimum

cumsum = torch.cumsum
def empty(shape, ctx=None):
    return torch.empty(shape, device=ctx)

def get_ctx(data):
    return data.device


def tile(data, reps):
    return data.repeat(reps)


def reshape(data, shape):
    return data.view(shape)
