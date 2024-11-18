import os
from typing import Optional, Tuple, List
from cuda import cuda, nvrtc, cudart


class NVRTCException(Exception):
    pass


class CompileException(Exception):
    pass


def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError("Unknown error type: {}".format(error))


def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


class _NVRTCProgram(object):
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
        print(source)
        print(name)
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
        print(num_headers, headers, include_names)
        self.program = checkCudaErrors(
            nvrtc.nvrtcCreateProgram(self.source_bytes, self.name_bytes, num_headers, headers, include_names)
        )
        self.method = method

    def __del__(self):
        if self.program:
            checkCudaErrors(nvrtc.destroyProgram(self.program))

    def get_source(self):
        return self.source

    def compile(self, num_options, options, named_expresions):
        # TODO: Setup error handling
        if named_expresions is not None:
            for name in named_expresions:
                checkCudaErrors(nvrtc.nvrtcAddNameExpression(self.program, name))
        checkCudaErrors(nvrtc.nvrtcCompileProgram(self.program, len(options), options))
        mapping = None
        if named_expresions:
            mapping = {}
            for name in named_expresions:
                mapping[name] = checkCudaErrors(nvrtc.nvrtcGetLoweredName(self.program, name))

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


def create_program(
    source: str | bytes,
    name: str | bytes,
    num_headers: int = 0,
    headers: Optional[Tuple[bytes] | List[bytes]] = None,
    include_names: Optional[Tuple[bytes] | List[bytes]] = None,
    method: str = "ptx",
):
    if os.path.isfile(source):
        with open(source, "r") as f:
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
