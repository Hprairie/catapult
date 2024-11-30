import ctypes
import torch
from cuda import cuda
from typing import List, TypeVar, Generic, Optional, overload, Callable, Union, Any, Protocol

from catapult.compiler.compiler import create_program, checkCudaErrors

from . import types
from .types import dtype


T = TypeVar("T")
R = TypeVar("R")


class KernelParams:
    """Represents a Kernel Params of a @jit'ed function"""

    _template_conversions = {
        types.int1: lambda arg: str(arg),
        types.int8: lambda arg: str(arg),
        types.int16: lambda arg: str(arg),
        types.int32: lambda arg: str(arg),
        types.int64: lambda arg: str(arg),
        types.float16: lambda arg: str(arg),
        types.float32: lambda arg: str(arg),
        types.float64: lambda arg: str(arg),
        types.bfloat16: lambda arg: str(arg),
        types.uint8: lambda arg: str(arg),
        types.uint16: lambda arg: str(arg),
        types.uint32: lambda arg: str(arg),
        types.uint64: lambda arg: str(arg),
        types.void: lambda arg: str(arg),
        int: lambda arg: str(arg),
        float: lambda arg: str(arg),
        str: lambda arg: str(arg),
        bool: lambda arg: str(arg).lower(),
    }

    _special_kernel_kwargs = ["stream", "smem"]

    def __init__(
        self,
        kernel_path: str,
        kernel_name: str,
        is_template: bool,
        template_params: Optional[List[str]],
        include: Optional[List[str]],
        method: Optional[str],
    ) -> None:
        self.kernel_path = kernel_path
        self.kernel_name = kernel_name
        self.is_template = is_template
        self.template_params = template_params
        self.headers = include
        if not isinstance(method, str):
            self.method = "ptx"
        else:
            self.method = method

        self.program = create_program(
            source=kernel_path, name=kernel_name, num_headers=0, headers=None, include_names=None, method=self.method
        )

    def __del__(self):
        del self.program
        return

    def _get_template(self, template_vals):
        if self.template_params is None:
            # TODO: Better error messaging
            raise ValueError(
                "No `template_params` were passed to catapult.jit, however **kwargs where passed to kernel."
            )
        template = []
        extra_includes = []
        for key in self.template_params:
            if key in self._special_kernel_kwargs:
                continue
            val = template_vals[key]
            if type(val) not in self._template_conversions:
                # TODO: Get better error handeling
                raise ValueError("NOT ALLOWABLE TYPE")
            template.append(self._template_conversions[type(val)](val))
            if isinstance(val, dtype) and val.include_files is not None:
                extra_includes += val.include_files

        return f"{self.kernel_name}<{', '.join(template)}>", extra_includes

    def get_compiled_kernel(self, options, template_vals):
        named_expression = None
        if template_vals is not None and template_vals:  # Check if not None and not empty
            named_expression, extra_includes = self._get_template(template_vals)
        num_options = len(options)
        compiled_code, mapping = self.program.compile(
            num_options=num_options, options=options, named_expression=named_expression
        )
        module = checkCudaErrors(cuda.cuModuleLoadData(compiled_code))
        kernel = checkCudaErrors(cuda.cuModuleGetFunction(module, self.program.get_name()))
        return kernel, mapping


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
        compile_options: Optional[List[str]] = None,
        debug: Optional[bool] = None,
        template_params: Optional[List[str]] = None,
        include: Optional[List[str]] = None,
        method: Optional[str] = "ptx",
    ) -> None:
        self.templated = template_params is not None
        self.kernel_params = KernelParams(
            kernel_path=kernel_path,
            kernel_name=kernel_name,
            is_template=self.templated,
            template_params=template_params,
            include=include,
            method=method,
        )
        self.debug = debug

        if not torch.cuda.is_initialized():
            torch.cuda.init()

        # Get compute capability and architecture argument
        self.cuDevice = checkCudaErrors(cuda.cuDeviceGet(torch.cuda.current_device()))
        self.major = checkCudaErrors(
            cuda.cuDeviceGetAttribute(
                cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, self.cuDevice
            )
        )
        self.minor = checkCudaErrors(
            cuda.cuDeviceGetAttribute(
                cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, self.cuDevice
            )
        )

        # Compile options need to be set after self.major and self.minor
        self.compile_options = self._get_options(compile_options)

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

    @staticmethod
    def _get_stream():
        stream = torch.cuda.current_stream()
        if stream.cuda_stream == 0:
            torch.cuda.set_stream(torch.cuda.Stream())
        stream = torch.cuda.current_stream()
        return stream

    def __del__(self):
        if self.kernel_params is not None:
            del self.kernel_params
        return

    def _get_options(self, compile_options):
        """
        Get compilation options for the CUDA kernel, handling GPU architecture specifications.

        Returns:
            List[bytes]: List of compilation options as bytes objects
        """
        # Start with default options if provided during initialization
        options = list(compile_options) if compile_options else []

        # Convert all string options to bytes if they aren't already
        options = [opt if isinstance(opt, bytes) else opt.encode("ascii") for opt in options]

        # Generate architecture argument based on compute capability
        arch_spec = f"sm_{self.major}{self.minor}"

        # Check if there's an existing architecture specification
        has_arch_spec = False
        for i, opt in enumerate(options):
            opt_str = opt.decode("ascii")
            if opt_str.startswith("--gpu-architecture="):
                # Update existing architecture specification
                current_arch = opt_str.split("=")[1]
                # Combine architectures if different
                if arch_spec not in current_arch:
                    new_archs = f"{current_arch},{arch_spec}"
                    options[i] = f"--gpu-architecture={new_archs}".encode("ascii")
                has_arch_spec = True
                break

        # Add new architecture specification if none exists
        if not has_arch_spec:
            options.append(f"--gpu-architecture={arch_spec}".encode("ascii"))

        # Add default options if their keys aren't already present
        default_opts = [b"--fmad=false"]
        for default_opt in default_opts:
            default_key = default_opt.split(b"=")[0]
            # Check if any existing option starts with this key
            if not any(opt.startswith(default_key) for opt in options):
                options.append(default_opt)

        return options

    # def __call__(self, *args, **kwargs):
    #     # TODO: Create better error
    #     raise RuntimeError("Not able to call kernel object")

    def run(self, *args, grid=None, thread_grid=None, warmup=None, **kwargs):
        # TODO: Make better errors
        if grid is None:
            raise ValueError("GRID IS NONE")
        if thread_grid is None:
            raise ValueError("THREAD GRID IS NONE")

        # TODO add caching
        kernel, mapping = self.kernel_params.get_compiled_kernel(options=self.compile_options, template_vals=kwargs)

        arg_values, arg_types = self._clean_values(args)

        checkCudaErrors(
            cuda.cuLaunchKernel(
                kernel,
                grid[0],
                grid[1],
                grid[2],
                thread_grid[0],
                thread_grid[1],
                thread_grid[2],
                kwargs.get("smem", 0),
                kwargs.get("stream", self._get_stream().cuda_stream),
                (arg_values, arg_types),
                0,
            )
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

        kernel = JITKernel(
            kernel_path=kernel_path,
            kernel_name=kernel_name,
            compile_options=compile_options,
            debug=debug,
            template_params=template_params,
            include=include,
            method=method,
        )

        def wrapper(*args: Any, **kwargs: Any) -> R:
            return func(kernel, *args, **kwargs)

        return wrapper

    return decorator
