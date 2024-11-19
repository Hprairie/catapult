import ctypes
import torch
from cuda import cuda
from catapult.compiler.compiler import create_program, checkCudaErrors
from typing import List, TypeVar, Generic, Optional, overload, Callable, Union, Any

T = TypeVar("T")


class KernelParams:
    """Represents a Kernel Params of a @jit'ed function"""

    def __init__(
        self,
        kernel_path: str,
        kernel_name: str,
        is_template: bool,
        template_params: List[str],
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
        template = []
        for key in self.template_params:
            template.append(str(template_vals[key]))
        return f"{self.kernel_name}<{', '.join(template)}>"

    def get_compiled_kernel(self, options, template_vals):
        named_expression = None
        if template_vals is not None:
            named_expression = self._get_template(template_vals)
        # if self.is_template or template_vals is not None:
        #     raise NotImplementedError("Compiling with named_expression is not currently enabled.")
        num_options = len(options)
        compiled_code, mapping = self.program.compile(
            num_options=num_options, options=options, named_expresion=named_expression
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
        compile_options: List[str],
        debug: bool,
        launch_metadata,
        template_params: List[str],
        include: List[str],
        method,
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
            new_stream = torch.cuda.Stream()
            torch.cuda.set_stream(new_stream)
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

    def run(self, *args, grid=None, thread_grid=None, warmup=None, **kwargs):
        # TODO: Make better errors
        if grid is None:
            raise ValueError("GRID IS NONE")
        if thread_grid is None:
            raise ValueError("THREAD GRID IS NONE")

        # TODO add caching
        kernel, mapping = self.kernel_params.get_compiled_kernel(options=self.compile_options, template_vals=kwargs)

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
    kernel_path: Optional[str] = None,
    kernel_name: Optional[str] = None,
    compile_options: Optional[List[str]] = None,
    method: Optional[str] = None,
    launch_metadata: Optional[Callable] = None,
    template_params: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
    debug: Optional[bool] = None,
) -> Callable[[T], JITKernel[T]]: ...


def jit(
    fn: Optional[T] = None,
    *,
    kernel_path: Optional[str] = None,
    kernel_name: Optional[str] = None,
    compile_options: Optional[List[str]] = None,
    method: Optional[str] = None,
    launch_metadata: Optional[Callable] = None,
    template_params: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
    debug: Optional[bool] = None,
) -> Union[JITKernel[T], Callable[[T], JITKernel[T]]]:
    def decorator(func: T) -> JITKernel[T]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            kernel = JITKernel(
                kernel_path=kernel_path,
                kernel_name=kernel_name,
                compile_options=compile_options,
                debug=debug,
                launch_metadata=launch_metadata,
                template_params=template_params,
                include=include,
                method=method,
            )
            return func(kernel, *args, **kwargs)

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator
