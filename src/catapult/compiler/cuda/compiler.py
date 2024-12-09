import os
from typing import Optional, Tuple, List
from cuda import cuda, nvrtc, cudart

from .errors import CompileException, NVRTCException, checkCudaErrors

from catapult.compiler.base import Compiler


class _NVRTCProgram(Compiler):
    def __init__(
        self,
        source: bytes,
        name: bytes,
        num_headers: int = 0,
        headers: Optional[Tuple[bytes] | List[bytes]] = None,
        include_names: Optional[Tuple[bytes] | List[bytes]] = None,
        method: str = "ptx",
    ):
        self.source_bytes = source
        self.name_bytes = name
        source = source.decode("UTF-8")
        name = name.decode("UTF-8")
        self.source = source
        self.name = name
        if num_headers < 0:
            raise CompileException(
                f"Error instantiating kernel: {self.name}",
                f"Value num_headers was passed < 0 and should be >= 0",
            )
        if num_headers > 0 and headers is None:
            raise CompileException(
                f"Error instantiating kernel: {self.name}",
                f"Value num_headers > 0, but headers is None type",
            )
        if num_headers > 0 and include_names is None:
            raise CompileException(
                f"Error instantiating kernel: {self.name}",
                f"Value num_headers > 0, but include_names is None type",
            )
        self.program = checkCudaErrors(
            nvrtc.nvrtcCreateProgram(self.source_bytes, self.name_bytes, num_headers, headers, include_names)
        )
        self.method = method

    def __del__(self):
        pass

    def get_source(self):
        return self.source

    def get_name(self):
        return self.name_bytes

    def compile(self, num_options, options, named_expression):
        # TODO: Setup error handling
        if named_expression is not None:
            if isinstance(named_expression, str):
                named_expression = bytes(named_expression, "utf-8")
            checkCudaErrors(nvrtc.nvrtcAddNameExpression(self.program, named_expression))
        checkCudaErrors(nvrtc.nvrtcCompileProgram(self.program, len(options), options))
        mapping = None
        if named_expression:
            self.name_bytes = checkCudaErrors(nvrtc.nvrtcGetLoweredName(self.program, named_expression))

        if self.method == "cubin":
            # TODO: Check if this works
            raise NotImplementedError("CUBIN NOT ALLOWED.")
            return checkCudaErrors(nvrtc.nvrtcGetCUBIN(self.program)), mapping
        elif self.method == "ptx":
            ptx_size = checkCudaErrors(nvrtc.nvrtcGetPTXSize(self.program))
            ptx = b" " * ptx_size
            checkCudaErrors(nvrtc.nvrtcGetPTX(self.program, ptx))
            return ptx, mapping
        else:
            raise CompileException(
                f"Error instantiating kernel: {self.name}",
                f"Unknown compilation method: {self.method}",
            )

