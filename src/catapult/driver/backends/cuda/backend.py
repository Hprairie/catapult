import os
from typing import Optional, Tuple, List

from cuda import cuda

from catapult.backends import Backend
from catapult.compiler import _NVRTCProgram
from catapult.compiler.cuda.errors import checkCudaErrors


from catapult.driver import GPUFramework


class CUDABackend(Backend):
    def __init__(self, framework: GPUFramework) -> None:
        if not isinstance(framework, GPUFramework):
            raise TypeError(f"Expected a GPUFramework when creating a CUDA Backend, but got {type(framework)}")
        self.framework = framework

    def get_compiler(
        source: str | bytes,
        name: str | bytes,
        calling_dir: str,
        num_headers: int = 0,
        headers: Optional[Tuple[bytes] | List[bytes]] = None,
        include_names: Optional[Tuple[bytes] | List[bytes]] = None,
        method: str = "ptx",
    ) -> _NVRTCProgram:
        if isinstance(source, str) and os.path.isfile(os.path.join(calling_dir, source)):
            with open(os.path.join(calling_dir, source), "r") as f:
                source = f.read()
        if isinstance(source, str):
            source = bytes(source, "utf-8")
        if isinstance(name, str):
            name = bytes(name, "utf-8")
        return _NVRTCProgram(
            source=source,
            name=name,
            num_headers=num_headers,
            headers=headers,
            include_names=include_names,
            method=method,
        )

    @classmethod
    def is_available() -> bool:
        raise NotImplementedError
    
    def launch_backend(self, kernel, grid, thread_grid, arg_values, arg_types, **kwargs) -> None:
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
                kwargs.get("smem", 0),
                kwargs.get("stream", self.framework.get_stream()),
                (arg_values, arg_types),
                0,
            )
        )
    