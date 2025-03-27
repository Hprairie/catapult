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
        include: Optional[List[str]] = None,
    ) -> None:
        self.templated = template_params is not None
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
        )
        self.cache = defaultdict(dict)

    def _get_signature(self, *args, **kwargs):
        """
        Extracts template parameter values from kwargs to create a unique signature for kernel caching.

        Args:
            *args: Variable positional arguments (unused).
            **kwargs: Keyword arguments that may contain template parameter values.

        Returns:
            list: List of template parameter values used for cache key generation.
        """
        constexpr_vals = []
        for key, val in kwargs.items():
            if key in self.compiler.template_params:
                constexpr_vals.append(val)
        return constexpr_vals

    def __call__(self, *args, **kwargs):
        raise KernelLaunchError(
            "JITKernel object can not be called directly and instead should be launched using a grid configuration.\n"
            "Example: kernel[(32, 1, 1), (256, 1, 1)](*args, **kwargs)\n"
            "         |      |_block dims   |_thread dims\n"
            "         |_kernel object"
            "\n"
            "If you are seeing this error, it means you are trying to call the kernel directly without a grid configuration."
        )

    def run(
        self,
        *args,
        grid: Optional[List[int]],
        thread_grid: Optional[List[int]],
        warmup: Optional[List[int]] = None,
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

        constexpr_vals = self._get_signature(*args, **kwargs)
        key = str(constexpr_vals)
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
            kernel_param=kernel_param,
            calling_dir=calling_dir,
            compile_options=compile_options,
            debug=debug,
            template_params=template_params,
            include=include,
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            return func(*args, **kwargs)

        wrapper.kernel = kernel

        return wrapper

    return decorator
