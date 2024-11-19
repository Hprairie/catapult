import torch
from functools import wraps


# Simple wrapper around torch.library.custom_op() for simplicity
def custom_op(name, mutates_args=(), device_types=None, schema=None):
    def decorator(fn):
        # Call torch.library.custom_op with the same arguments
        decorated_fn = torch.library.custom_op(
            name, mutates_args=mutates_args, device_types=device_types, schema=schema
        )(fn)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            return decorated_fn(*args, **kwargs)

        return wrapper

    return decorator


# Simple wrapper around torch.register_autograd() for simplicity
def register_autograd(backward_fn, setup_context_fn):
    def decorator(func):
        registration_done = False

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal registration_done
            result = func(*args, **kwargs)

            # Only register once
            if not registration_done:
                func.register_autograd(backward_fn, setup_context=setup_context_fn)
                registration_done = True
            return result

        return wrapper

    return decorator
