from catapult.compiler.compiler import create_program
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

    def __getitem__(self, grid) -> T:
        return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)


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

    @staticmethod
    def _clean_values(*args):
        pass

    def run(self, grid=None, warmup=None, *args, **kwargs):
        pass


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
