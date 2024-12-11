import os
from typing import Optional, Tuple, List

from cuda import cuda

from catapult.driver import Backend
from catapult.compiler import _NVRTCProgram
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
        calling_dir: str,
        device: int,
        compile_options: List[bytes] | None = None,
        num_headers: int = 0,
        headers: Optional[Tuple[bytes] | List[bytes]] = None,
        include_names: Optional[Tuple[bytes] | List[bytes]] = None,
        template_params: Optional[List[str]] = None,
        method: str = "ptx",
    ) -> _NVRTCProgram:
        if isinstance(source, str) and os.path.isfile(os.path.join(calling_dir, source)):
            with open(os.path.join(calling_dir, source), "r") as f:
                source = f.read()
        if isinstance(source, str):
            source = bytes(source, "utf-8")
        if isinstance(name, str):
            name = bytes(name, "utf-8")
        # TODO: Add type checking for bytes for compile_options

        if method is None:
            method = "ptx"

        return _NVRTCProgram(
            source=source,
            name=name,
            device=device,
            num_headers=num_headers,
            compile_options=compile_options,
            headers=headers,
            include_names=include_names,
            template_params=template_params,
            method=method,
        )

    @staticmethod
    def is_available() -> bool:
        raise NotImplementedError
    
    def launch_backend(self, framework, kernel, grid, thread_grid, arg_values, arg_types, **kwargs) -> None:
        # TODO: This needs to be cleaned up
        checkCudaErrors(
            cuda.cuLaunchKernel(
                kernel,
                grid[0],
                grid[1],
                grid[2],
                thread_grid[0],
                thread_grid[1],
                thread_grid[2],
                int(kwargs.get("smem", 0)),
                int(kwargs.get("stream", framework.get_stream())),
                (arg_values, arg_types),
                0,
            )
        )
    