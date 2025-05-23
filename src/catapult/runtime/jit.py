import os
import functools
import inspect
from collections import defaultdict
from typing import List, TypeVar, Generic, Optional, overload, Callable, Union, Any, Protocol

from .build import get_driver
from .errors import KernelLaunchError


T = TypeVar("T")
R = TypeVar("R")


class KernelInterface(Generic[T]):
    """Interface for a @jit'ed function"""

    run: T

    def __getitem__(self, grid) -> T:
        if grid is None:
            raise KernelLaunchError(
                "Attempting to launch kernel without passing a grid configuration.\n"
                "Example: kernel[(32, 1, 1), (256, 1, 1)](*args, **kwargs)\n"
                "         |      |_block dims   |_thread dims\n"
                "         |_kernel object"
                "To use Python interface to launch please follow the example above. To default to the C++ interface please call the kernel directly.\n"
                "Exampele: kernel(*args, **kwargs)\n\n"
            )

        if not isinstance(grid, tuple) or len(grid) != 2 or len(grid[0]) != 3 or len(grid[1]) != 3:
            raise KernelLaunchError(
                "Grid configuration must be a tuple of (block_dims, thread_dims).\n"
                "Example: kernel[(32, 1, 1), (256, 1, 1)](*args, **kwargs)"
            )

        return lambda *args, **kwargs: self.run(*args, grid=grid[0], thread_grid=grid[1], warmup=False, **kwargs)


class JITKernel(KernelInterface[T]):
    """
    Just-In-Time compilation handler for GPU kernels.

    This class manages the compilation, caching, and execution of GPU kernels. It supports
    template parameters, custom compilation options, and maintains a device-specific cache
    of compiled kernels to avoid recompilation of previously used configurations.

    The kernel must be launched using square bracket syntax with grid configuration:
    kernel[(block_dims), (thread_dims)](*args, **kwargs)

    Attributes:
        templated (bool): Indicates if the kernel uses template parameters.
        template_kernel (bool): Indicates if the kernel name uses template parameters.
        driver: Backend driver instance for GPU operations.
        compiler: Compiler instance for the specific kernel.
        cache (defaultdict): Device-specific cache of compiled kernels.
    """

    def __init__(
        self,
        kernel_path: str,
        kernel_name: str,
        kernel_param: Optional[str],
        calling_dir: str,
        compile_options: Optional[List[str]] = None,
        debug: Optional[bool] = None,
        template_params: Optional[List[str]] = None,
        template_kernel: Optional[List[str]] = None,
        include: Optional[List[str]] = None,
        disable_stream: Optional[bool] = None,
    ) -> None:
        self.templated = template_params is not None
        self.template_kernel = template_kernel is not None
        self.driver = get_driver()

        # TODO: Add debugging option

        self.compiler = self.driver.backend.get_compiler(
            source=kernel_path,
            name=kernel_name,
            kernel_param=kernel_param,
            calling_dir=calling_dir,
            device=self.driver.framework.get_device(),
            compile_options=compile_options,
            include_names=include,
            template_params=template_params,
            template_kernel=template_kernel,
        )
        self.cache = defaultdict(dict)
        self.disable_stream = disable_stream

    def _get_signature(self, *args, **kwargs):
        """
        Extracts template parameter values from kwargs to create a unique signature for kernel caching.

        Args:
            *args: Variable positional arguments (unused).
            **kwargs: Keyword arguments that may contain template parameter values.

        Returns:
            list: List of template parameter values used for cache key generation.
        """
        param_vals = []
        kernel_vals = []
        
        if hasattr(self.compiler, "template_params") and self.compiler.template_params:
            for key in self.compiler.template_params:
                if key in kwargs:
                    param_vals.append(kwargs[key])
                    
        if hasattr(self.compiler, "template_kernel") and self.compiler.template_kernel:
            for key in self.compiler.template_kernel:
                if key in kwargs:
                    kernel_vals.append(kwargs[key])
                    
        return (tuple(param_vals), tuple(kernel_vals))

    def __call__(self, *args, **kwargs):
        return self.run(*args, grid=None, thread_grid=None, warmup=None, **kwargs)

    def run(
        self,
        *args,
        grid: Optional[List[int]],
        thread_grid: Optional[List[int]],
        warmup: Optional[bool] = None,
        **kwargs,
    ):
        """
        Executes the GPU kernel with the specified grid configuration and arguments.

        This method handles kernel compilation (if needed), caching, and launching.
        It maintains a cache of compiled kernels based on template parameters to avoid
        recompilation of previously used kernel configurations.

        Args:
            *args: Positional arguments to pass to the kernel.
            grid (List[int]): Block dimensions for the CUDA grid (x, y, z).
            thread_grid (List[int]): Thread dimensions for each block (x, y, z).
            warmup (Optional[List[int]], optional): Number of warmup iterations.
            **kwargs: Keyword arguments, including template parameters.

        Returns:
            None: The kernel execution is asynchronous.
        """
        device = self.driver.framework.get_device()

        # Add disable_stream to kwargs if it's set
        if self.disable_stream:
            kwargs["disable_stream"] = True

        signature = self._get_signature(*args, **kwargs)
        key = str(signature)
        kernel = self.cache[device].get(key, None)

        if kernel is None:
            self.compiler.compile(template_vals=kwargs)
            kernel, _ = self.compiler.get_kernel()
            self.cache[device][key] = kernel

        arg_values, arg_types = self.driver.framework.clean_values(args)

        self.driver.backend.launch_backend(
            self.driver.framework, kernel, grid, thread_grid, arg_values, arg_types, **kwargs
        )

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
    kernel_param: Optional[str] = None,
    compile_options: Optional[List[str]] = None,
    template_params: Optional[List[str]] = None,
    template_kernel: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
    debug: Optional[bool] = None,
) -> Callable[[Callable[..., R]], KernelFunction[R]]: ...


def jit(
    fn: Optional[T] = None,
    kernel_path: Optional[str] = None,
    kernel_name: Optional[str] = None,
    kernel_param: Optional[str] = None,
    compile_options: Optional[List[str]] = None,
    template_params: Optional[List[str]] = None,
    template_kernel: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
    debug: Optional[bool] = None,
    disable_stream: Optional[bool] = None,
) -> Union[KernelFunction[R], Callable[[Callable[..., R]], KernelFunction[R]]]:
    """
    JIT decorator for defining CUDA kernels with optional template and compile parameters.
    Requires explicit parameters.

    Args:
        kernel_path (str): Path to the CUDA kernel file.
        kernel_name (str): Name of the kernel function in the file.
        kernel_param (str, optional): Parameter type for the kernel.
        compile_options (List[str], optional): Additional options for NVRTC compiler.
        template_params (List[str], optional): Template parameters for the kernel parameter.
        template_kernel (List[str], optional): Template parameters for the kernel name.
        include (List[str], optional): Additional include paths for the kernel.
        debug (bool, optional): Enables debug mode for the kernel.
        disable_stream (bool, optional): Disables stream usage for kernel execution.

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
            kernel_param=kernel_param,
            calling_dir=calling_dir,
            compile_options=compile_options,
            debug=debug,
            template_params=template_params,
            template_kernel=template_kernel,
            include=include,
            disable_stream=disable_stream,
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            return func(*args, **kwargs)

        wrapper.kernel = kernel

        return wrapper

    return decorator
