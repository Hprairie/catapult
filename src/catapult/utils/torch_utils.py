import torch
from functools import wraps


def custom_op(
    name,
    mutates_args=(),
    device_types=None,
    schema=None,
    forward_fn=None,
    backward_fn=None,
    setup_context=None,
    register_fake=None,
):
    """Wrapper around PyTorch custom_op interface to be slightly cleaner with a single hub for autograd and torch.compile compatibility."""

    def decorator(fn):
        # Create custom op using PyTorch's API
        decorated_fn = torch.library.custom_op(
            name, mutates_args=mutates_args, device_types=device_types, schema=schema
        )(fn)

        # Register autograd if backward_fn is provided
        if backward_fn is not None:
            torch.library.register_autograd(name, backward_fn, setup_context=setup_context)

        # Register fake tensor if provided
        if register_fake is not None:
            torch.library.register_fake(name)(register_fake)

        return decorated_fn

    return decorator
