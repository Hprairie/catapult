import torch
from typing import Callable, List, Optional


def custom_op(
    name: str,
    mutates_args: tuple = (),
    device_types: Optional[List[str]] = None,
    schema: Optional[str] = None,
    backward_fn: Optional[Callable] = None,
    setup_context: Optional[Callable] = None,
    register_fake: Optional[Callable] = None,
) -> Callable:
    """Creates a custom PyTorch operator with automated registration of autograd and compilation features.

    This decorator simplifies the process of creating custom PyTorch operations by providing
    a unified interface for registering forward/backward operations, device compatibility,
    and compilation support.

    Args:
        name (str): The name of the custom operator.
        mutates_args (tuple): Indices of arguments that are mutated by the operation. Defaults to ().
        device_types (List[str], optional): List of supported device types (e.g., ["CPU", "CUDA"]).
        schema (str, optional): The operator's schema string in TorchScript format.
        backward_fn (Callable, optional): The backward (gradient) implementation function.
        setup_context (Callable, optional): Function to set up autograd context.
        register_fake (Callable, optional): Function to register fake tensor support for compilation.

    Returns:
        Callable: A decorated function that implements the custom operator.

    Example:
        ```python
        @custom_op(
            name="myproject::custom_add",
            device_types=["CPU", "CUDA"],
            schema="(Tensor x, Tensor y) -> Tensor"
        )
        def custom_add(x, y):
            return x + y
        ```
    """

    def decorator(fn):
        # Create custom op using PyTorch's API
        decorated_fn = torch.library.custom_op(
            name, mutates_args=mutates_args, device_types=device_types, schema=schema
        )(fn)

        if backward_fn is not None:
            torch.library.register_autograd(name, backward_fn, setup_context=setup_context)

        if register_fake is not None:
            torch.library.register_fake(name)(register_fake)

        decorated_fn.kernel = fn.kernel

        return decorated_fn

    return decorator
