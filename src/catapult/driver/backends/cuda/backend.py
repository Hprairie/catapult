import os
from typing import Optional, Tuple, List, Union

from cuda import cuda

from catapult.driver import Backend
from catapult.compiler import _NVRTCProgram
from catapult.compiler.cuda.nvcc import _NVCCProgram  # TODO: Fix this
from catapult.compiler.cuda.errors import checkCudaErrors


class CUDABackend(Backend):

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_name() -> str:
        return "cuda"

    @staticmethod
    def get_compiler(
        source: str | bytes,
        name: str | bytes,
        kernel_param: str | bytes,
        calling_dir: str,
        device: int,
        compile_options: List[bytes] | None = None,
        num_headers: Optional[int] = 0,
        headers: Optional[Tuple[bytes] | List[bytes]] = None,
        include_names: Optional[Tuple[bytes] | List[bytes]] = None,
        template_params: Optional[List[str]] = None,
        template_kernel: Optional[List[str]] = None,
    ) -> Union[_NVCCProgram, _NVRTCProgram]:

        if isinstance(source, str) and os.path.isfile(os.path.join(calling_dir, source)):
            with open(os.path.join(calling_dir, source), "r") as f:
                source = f.read()
        if isinstance(source, str):
            source = bytes(source, "utf-8")
        if isinstance(name, str):
            name = bytes(name, "utf-8")
        if isinstance(kernel_param, str):
            kernel_param = bytes(kernel_param, "utf-8")

        return _NVCCProgram(
            source=source,
            name=name,
            kernel_param=kernel_param,
            device=device,
            compile_options=compile_options,
            num_headers=num_headers,
            headers=headers,
            include_names=include_names,
            template_params=template_params,
            template_kernel=template_kernel,
        )

    @staticmethod
    def is_available() -> bool:
        raise NotImplementedError

    def launch_backend(self, framework, kernel, grid, thread_grid, arg_values, arg_types, **kwargs) -> None:
        """
        Launch a CUDA kernel.

        Args:
            framework: The framework object providing context (e.g., stream)
            kernel: The kernel function (from pybind module) or function pointer (for NVRTC)
            grid: Grid dimensions (3-tuple)
            thread_grid: Thread block dimensions (3-tuple)
            arg_values: Kernel arguments values
            arg_types: Kernel argument types
            **kwargs: Additional arguments
        """
        try:
            stream = framework.get_stream() if hasattr(framework, "get_stream") else None
            use_stream = stream is not None and not kwargs.get("disable_stream", False)
            use_grid = grid is not None and thread_grid is not None

            if hasattr(kernel, "CATAPULT_NAME"):
                base_name = kernel.CATAPULT_NAME
                func_name = base_name
            else:
                raise RuntimeError("Kernel Object Corrupted - Missing CATAPULT_NAME")

            if use_grid and use_stream:
                func_name += "_stream_grid"
                arg_values = arg_values + (
                    grid[0],
                    grid[1],
                    grid[2],
                    thread_grid[0],
                    thread_grid[1],
                    thread_grid[2],
                    stream,
                )
            elif use_grid:
                func_name += "_grid"
                arg_values = arg_values + (grid[0], grid[1], grid[2], thread_grid[0], thread_grid[1], thread_grid[2])
            elif use_stream:
                func_name += "_stream"
                arg_values = arg_values + (stream,)

            if hasattr(kernel, func_name):
                kernel_func = getattr(kernel, func_name)
                kernel_func(*arg_values)
            else:
                raise RuntimeError(f"Kernel function '{func_name}' not found in kernel object.")
        except Exception as e:
            raise RuntimeError(f"Error launching kernel: {str(e)}")
