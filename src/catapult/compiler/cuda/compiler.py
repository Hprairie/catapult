from typing import Optional, Tuple, List
from cuda import cuda, nvrtc

from .errors import CompileException, NVRTCException, checkCudaErrors

from catapult.compiler.base import Compiler
from catapult.runtime.types import dtype


class _NVRTCProgram(Compiler):
    def __init__(
        self,
        source: bytes,
        name: bytes,
        device: int,
        compile_options: Optional[List[bytes]] = None,
        num_headers: Optional[int] = None,
        headers: Optional[Tuple[bytes] | List[bytes]] = None,
        include_names: Optional[Tuple[bytes] | List[bytes]] = None,
        template_params: Optional[List[str]] = None,
        method: Optional[str] = "ptx",
    ):
        if not isinstance(source, bytes):
            raise CompileException(
                f"Error instantiaing NVRTC kernel Compiler.",
                f"Value source was passed with ({type(source)}) when it should be of type (bytes).",
            )
        if not isinstance(name, bytes):
            raise CompileException(
                f"Error instantiaing NVRTC kernel Compiler.",
                f"Value name was passed with ({type(name)}) when it should be of type (bytes).",
            )
        self.source_bytes = source
        self.name_bytes = name
        self.source = source.decode("UTF-8")
        self.name = name.decode("UTF-8")
        if num_headers is not None and num_headers < 0:
            raise CompileException(
                f"Error instantiating kernel: {self.name}",
                f"Value num_headers was passed < 0 and should be >= 0",
            )
        if num_headers is not None and num_headers > 0 and headers is None:
            raise CompileException(
                f"Error instantiating kernel: {self.name}",
                f"Value num_headers > 0, but headers is None type",
            )
        if num_headers is not None and num_headers > 0 and include_names is None:
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
        self.template_params = template_params

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

    def get_source(self) -> bytes:
        return self.source_bytes

    def get_name(self) -> bytes:
        if not isinstance(self.name_bytes, bytes):
            raise NVRTCException(
                f"Error in NVRTC Program get_name(): Expected name_bytes to be of type 'bytes' but got '{type(self.name_bytes).__name__}'. "
                "This may indicate corruption of the internal program name or improper initialization."
            )
        return self.name_bytes

    def compile(self, template_vals) -> None:
        # TODO: Setup error handling
        named_expression = None
        if len(template_vals):
            named_expression, extra_includes = self._create_template_string(template_vals)
            named_expression = bytes(named_expression, "utf-8")
            checkCudaErrors(nvrtc.nvrtcAddNameExpression(self.program, named_expression), self.program)
        checkCudaErrors(
            nvrtc.nvrtcCompileProgram(self.program, len(self.compile_options), self.compile_options), self.program
        )
        mapping = None
        if named_expression:
            # TODO: Check if this is a good way of doing this
            self.name_bytes = checkCudaErrors(nvrtc.nvrtcGetLoweredName(self.program, named_expression), self.program)

        ptx_size: int = checkCudaErrors(nvrtc.nvrtcGetPTXSize(self.program), self.program)
        ptx = b" " * ptx_size
        checkCudaErrors(nvrtc.nvrtcGetPTX(self.program, ptx), self.program)
        self.compiled_program = ptx
        self.mapping = mapping
        return

    def _get_options(self, compile_options):
        """
        Get compilation options for the CUDA kernel, handling GPU architecture specifications.

        Returns:
            List[bytes]: List of compilation options as bytes objects
        """
        options = list(compile_options) if compile_options else []
        options = [opt if isinstance(opt, bytes) else opt.encode("ascii") for opt in options]

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

        if not has_arch_spec:
            options.append(f"--gpu-architecture={arch_spec}".encode("ascii"))

        # Add default options if their keys aren't already present
        default_opts = [
            b"--fmad=false",
        ]
        for default_opt in default_opts:
            default_key = default_opt.split(b"=")[0]
            # Check if any existing option starts with this key
            if not any(opt.startswith(default_key) for opt in options):
                options.append(default_opt)

        return options

    def _create_template_string(self, template_vals):
        if self.template_params is None:
            raise ValueError(
                "Template parameters are required but none were provided in the @catapult.jit decorator.\n"
                "Example usage:\n"
                "@catapult.jit(\n"
                "    kernel_path='example_template.cuh',\n"
                "    kernel_name='example_kernel_name',\n"
                "    template_params=['N']  # List the template parameters in order\n"
                ")\n"
                f"template_params is None but Received kwargs: {list(template_vals.keys())}"
            )
        template = []
        extra_includes = []
        for key in self.template_params:
            if key in self._special_kernel_kwargs:
                continue
            val = template_vals[key]

            # Verbose error message for unsupported types
            if type(val) not in self._template_conversions:
                type_groups = {"Python built-in types": [], "catapult.types": []}

                for allowed_type in self._template_conversions.keys():
                    if allowed_type.__module__ == "builtins":
                        type_groups["Python built-in types"].append(allowed_type.__name__)
                    else:
                        type_groups["catapult.types"].append(allowed_type.__name__)

                error_msg = [
                    f"Template parameter '{key}' has unsupported type '{type(val).__name__}'.",
                    "Allowed types are:",
                ]
                for group_name, types in type_groups.items():
                    if types:
                        error_msg.append(f"  * {group_name}: {', '.join(sorted(types))}")

                raise ValueError("\n".join(error_msg))

            template.append(self._template_conversions[type(val)](val))
            if isinstance(val, dtype) and val.include_files is not None:
                extra_includes += val.include_files

        return f"{self.name}<{', '.join(template)}>", extra_includes

    def get_kernel(self):
        if self.compiled_program is None:
            raise ValueError(
                f"Error accessing kernel '{self.name}': Kernel has not been compiled yet.\n"
                "The compile() method must be called before attempting to get the kernel.\n"
                "This usually happens automatically when the kernel is called with parameters.\n"
                "If you're seeing this error, it may indicate a problem with template parameter resolution "
                "or an issue in the compilation pipeline."
            )
        module = checkCudaErrors(cuda.cuModuleLoadData(self.compiled_program))
        kernel = checkCudaErrors(cuda.cuModuleGetFunction(module, self.name_bytes))
        return kernel, self.mapping
