import torch
from functools import wraps


def custom_op(name, mutates_args=(), device_types=None, schema=None, backward_fn=None, setup_context_fn=None):
    def decorator(fn):
        # Create custom op using PyTorch's API
        decorated_fn = torch.library.custom_op(
            name, mutates_args=mutates_args, device_types=device_types, schema=schema
        )(fn)

        # Register autograd if backward_fn is provided
        if backward_fn is not None:
            torch.library.register_autograd(name, backward_fn, setup_context=setup_context_fn)

        return decorated_fn

    return decorator
