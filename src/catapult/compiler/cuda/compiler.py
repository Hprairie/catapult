import os
from typing import Optional, Tuple, List
from cuda import cuda, nvrtc, cudart

from .errors import CompileException, NVRTCException, checkCudaErrors

from catapult.compiler.base import Compiler
from catapult.runtime.types import dtype
from catapult.driver import Framework


class _NVRTCProgram(Compiler):
    def __init__(
        self,
        source: bytes,
        name: bytes,
        device: int,
        compile_options: List[bytes] = None,
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
        self.device = device
        self.compiled_program = None
        self.mapping = None
        self.named_expression = {}

        # Get compute capability and architecture argument
        self.cuDevice = checkCudaErrors(cuda.cuDeviceGet(device))
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


    def __del__(self):
        pass

    def get_source(self):
        return self.source

    def get_name(self):
        return self.name_bytes

    def compile(self, num_options, options, template_vals):
        # TODO: Setup error handling
        named_expression = None
        if template_vals is not None:
            named_expression, extra_includes = self._create_template_string(template_vals)
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
            self.compiled_program = ptx
            self.mapping = mapping
            return
        else:
            raise CompileException(
                f"Error instantiating kernel: {self.name}",
                f"Unknown compilation method: {self.method}",
            )
    
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
    
    def _create_template_string(self, template_vals):
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


    def get_kernel(self):
        if self.compiled_code is None or self.mapping is None:
            raise ValueError("Attemtping to get kernel before compiling the program?")
        module = checkCudaErrors(cuda.cuModuleLoadDataEx(self.compiled_code))
        kernel = checkCudaErrors(cuda.cuModuleGetFunction(module, self.name_bytes))
        return kernel, self.mapping
