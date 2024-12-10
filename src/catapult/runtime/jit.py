import os
import functools
import inspect
import ctypes
from collections import defaultdict
from typing import List, TypeVar, Generic, Optional, overload, Callable, Union, Any, Protocol

from .build import get_driver


T = TypeVar("T")
R = TypeVar("R")



class KernelInterface(Generic[T]):
    """Interface for a @jit'ed function"""

    run: T

    def __getitem__(self, grid) -> T:
        return lambda *args, **kwargs: self.run(*args, grid=grid[0], thread_grid=grid[1], warmup=False, **kwargs)


class JITKernel(KernelInterface[T]):

    def __init__(
        self,
        kernel_path: str,
        kernel_name: str,
        calling_dir: str,
        compile_options: Optional[List[str]] = None,
        debug: Optional[bool] = None,
        template_params: Optional[List[str]] = None,
        include: Optional[List[str]] = None,
        method: Optional[str] = "ptx",
    ) -> None:
        self.templated = template_params is not None
        self.driver = get_driver()

        # TODO: Initialize the compiler object
        self.compiler = self.driver.backend.get_compiler()
        self.cache = defaultdict(dict)

    @staticmethod
    def _clean_values(args):
        """
        Prepares arguments for CUDA kernel launch with proper C types using a dictionary-based approach.

        Args:
            *args: Variable number of arguments of different types

        Returns:
            tuple: (arg_values, arg_types) for cuLaunchKernel
        """

        # Dictionary mapping Python types to their corresponding ctype handlers

        TYPE_HANDLERS = {
            torch.Tensor: lambda x: (x.data_ptr(), ctypes.c_void_p),
            bool: lambda x: (ctypes.c_bool(x), ctypes.c_bool),
            int: lambda x: (ctypes.c_size_t(x), ctypes.c_size_t),
            float: lambda x: (ctypes.c_float(x), ctypes.c_float),
        }

        def get_ctype_and_value(arg):
            # Look up the handler for this type
            handler = TYPE_HANDLERS.get(type(arg))
            if handler is None:
                raise TypeError(f"Unsupported argument type: {type(arg)}")
            return handler(arg)

        # Process all arguments
        processed_args = [get_ctype_and_value(arg) for arg in args]

        # Split into values and types
        arg_values = tuple(value for value, _ in processed_args)
        arg_types = tuple(type_ for _, type_ in processed_args)

        return arg_values, arg_types

    def _get_signature(self, *args, **kwargs):
        constexpr_vals = []
        for key, val in kwargs.items():
            if key in self.kernel_params.template_params:
                constexpr_vals.append(val)
        return constexpr_vals

    def __call__(self, *args, **kwargs):
        # TODO: Create better error
        raise RuntimeError("Not able to call kernel object")

    def run(self, *args, grid=None, thread_grid=None, warmup=None, **kwargs):
        # TODO: Make better errors
        if grid is None:
            raise ValueError("GRID IS NONE")
        if thread_grid is None:
            raise ValueError("THREAD GRID IS NONE")

        device = self.driver.framework.get_device()

        constexpr_vals = self._get_signature(*args, **kwargs)
        key = str(constexpr_vals)
        kernel = self.cache[device].get(key, None)

        if kernel is None:
            self.compiler.compile(options=self.compile_options, template_vals=kwargs)
            self.cache[device][key] = kernel

        # TODO: Abstract this to the driver's framework
        arg_values, arg_types = self._clean_values(args)

        self.driver.backend.launch_backend(kernel, grid, thread_grid, arg_values, arg_types, **kwargs)

        return


class KernelFunction(Protocol[R]):
    def __call__(self, kernel: JITKernel, *args: Any, **kwargs: Any) -> R: ...


# ---------------------------
# JIT decorator
# ---------------------------


@overload
def jit(fn: Callable[..., R]) -> KernelFunction[R]: ...


@overload
def jit(
    *,
    kernel_path: str,
    kernel_name: str,
    compile_options: Optional[List[str]] = None,
    method: Optional[str] = None,
    template_params: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
    debug: Optional[bool] = None,
) -> Callable[[Callable[..., R]], KernelFunction[R]]: ...


def jit(
    fn: Optional[T] = None,
    kernel_path: Optional[str] = None,
    kernel_name: Optional[str] = None,
    compile_options: Optional[List[str]] = None,
    method: Optional[str] = None,
    template_params: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
    debug: Optional[bool] = None,
) -> Union[KernelFunction[R], Callable[[Callable[..., R]], KernelFunction[R]]]:
    """
    JIT decorator for defining CUDA kernels with optional template and compile parameters.
    Requires explicit parameters.

    Args:
        kernel_path (str): Path to the CUDA kernel file.
        kernel_name (str): Name of the kernel function in the file.
        compile_options (List[str], optional): Additional options for NVRTC compiler.
        method (str, optional): Compilation method, default is "ptx".
        template_params (List[str], optional): Template parameters for the kernel.
        include (List[str], optional): Additional include paths for the kernel.
        debug (bool, optional): Enables debug mode for the kernel.

    Returns:
        Callable[[Callable[..., R]], KernelFunction[R]]: The decorated function.
    """

    def decorator(func: Callable[..., R]) -> KernelFunction[R]:
        if kernel_path is None or kernel_name is None:
            # TODO: Create better errors
            raise ValueError("kernel_path or kernel_name are not set.")

        calling_script = os.path.abspath(inspect.stack()[1].filename)
        calling_dir = os.path.dirname(calling_script)

        kernel = JITKernel(
            kernel_path=kernel_path,
            kernel_name=kernel_name,
            calling_dir=calling_dir,
            compile_options=compile_options,
            debug=debug,
            template_params=template_params,
            include=include,
            method=method,
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            return func(*args, **kwargs)

        wrapper.kernel = kernel

        return wrapper

    return decorator
