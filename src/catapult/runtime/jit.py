import ctypes
import torch
from cuda import cuda, driver
from catapult.compiler.compiler import create_program, checkCudaErrors
from typing import List, TypeVar, Generic, Optional, overload, Callable, Union

T = TypeVar("T")


class KernelParams:
    """Represents a Kernel Params of a @jit'ed function"""

    def __init__(
        self,
        kernel_path: str,
        kernel_name: str,
        is_template: bool,
        template_params: List[str],
        headers: Optional[List[str]],
        method: Optional[str],
    ) -> None:
        self.kernel_path = kernel_path
        self.kernel_name = kernel_name
        self.is_template = is_template
        self.template_params = template_params
        self.headers = headers
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

    def get_compiled_kernel(self, options, named_expression):
        if options is not None:
            raise NotImplementedError("Compiling with options is not currently enabled.")
        if named_expression is not None:
            raise NotImplementedError("Compiling with named_expression is not currently enabled.")
        num_options = len(options)
        return self.program.compile(num_options=num_options, options=options, named_expresions=named_expression)


class KernelInterface(Generic[T]):
    """Interface for a @jit'ed function"""

    run: T

    def __getitem__(self, grid, thread_grid) -> T:
        return lambda *args, **kwargs: self.run(grid=grid, thread_grid=thread_grid, warmup=False, *args, **kwargs)


class JITKernel(KernelInterface[T]):
    def __init__(
        self,
        kernel_path,
        kernel_name,
        compile_options,
        debug,
        launch_metadata,
        template_params,
        headers,
        method,
    ) -> None:
        self.templated = template_params is not None
        self.kernel_params = KernelParams(
            kernel_path=kernel_path,
            kernel_name=kernel_name,
            is_template=self.templated,
            template_params=template_params,
            headers=headers,
            method=method,
        )
        self.debug = debug
        self.compile_options = compile_options
        self.launch_metadata = launch_metadata

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

    @staticmethod
    def _clean_values(*args):
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
            new_stream = torch.cuda.Stream()
            torch.cuda.set_stream(new_stream)
        stream = torch.cuda.current_stream()
        return stream

    def __del__(self):
        del self.kernel_params
        return

    def run(self, grid=None, thread_grid=None, warmup=None, *args, **kwargs):

        if kwargs is not None:
            raise NotImplementedError("Passing template values as kwargs is not supported currently")
        # TODO: Make better errors
        if grid is None:
            raise ValueError("GRID IS NONE")
        if thread_grid is None:
            raise ValueError("THREAD GRID IS NONE")

        # TODO add caching
        kernel = self.kernel_params.get_compiled_kernel(options=None, named_expression=None)

        stream = self._get_stream()

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
                0,
                stream.cuda_stream,
                (arg_values, arg_types),
                0,
            )
        )
        return


# ---------------------------
# JIT decorator
# ---------------------------


@overload
def jit(fn: T) -> JITKernel[T]: ...


@overload
def jit(
    *,
    method=None,
    launch_metadata: Optional[Callable] = None,
    debug: Optional[bool] = None,
) -> Callable[[T], JITKernel[T]]: ...


def jit(
    fn: T = None,
    kernel_path: Optional[str] = None,
    kernel_name: Optional[str] = None,
    compile_options: Optional[List[str]] = None,
    method=None,
    launch_metadata: Optional[Callable] = None,
    template_params: Optional[List[str]] = None,
    headers: Optional[List[str]] = None,
    debug: Optional[bool] = None,
) -> Union[JITKernel[T], Callable[[T], JITKernel[T]]]:
    def decorator(fn: T) -> JITKernel[T]:
        def wrapper(*args, **kwargs):
            kernel = JITKernel(
                kernel_path=kernel_path,
                kernel_name=kernel_name,
                compile_options=compile_options,
                debug=debug,
                launch_metadata=launch_metadata,
                template_params=template_params,
                headers=headers,
                method=method,
            )
            return fn(kernel, *args, **kwargs)

        return wrapper

    if fn is not None:
        return decorator(fn)

    else:
        return decorator
