from cuda import cuda, nvrtc, cudart


class NVRTCException(Exception):
    def __init__(self, message, log=None):
        self.message = message
        self.log = log
        super().__init__(self.format_message())
    
    def format_message(self):
        if self.log:
            return f"{self.message}\n\nDumping NVRTC Log:\n\n{self.log}"
        return self.message

class CompileException(Exception):
    def __init__(self, message, details=None):
        self.message = message
        self.details = details
        super().__init__(self.format_message())
    
    def format_message(self):
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message

def get_nvrtc_program_log(program):
    try:
        log_size = nvrtc.nvrtcGetProgramLogSize(program)[1]
        program_log = b" " * log_size
        nvrtc.nvrtcGetProgramLog(program, program_log)
        return program_log.decode('utf-8')
    except:
        return None

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


def checkCudaErrors(result, program=None):
    if result[0].value:
        error_code = result[0].value
        error_name = _cudaGetErrorEnum(result[0])
        
        if isinstance(result[0], nvrtc.nvrtcResult):
            raise NVRTCException(
                f"NVRTC error occurred: {error_name} (code={error_code})",
                log=get_nvrtc_program_log(program) if program is not None else None,
            )
        else:
            raise RuntimeError(f"CUDA error: {error_name} (code={error_code})")

    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]
