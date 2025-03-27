from typing import Optional, Tuple, List
import os
import tempfile
import subprocess

from cuda import cuda

from .errors import CompileException, checkCudaErrors

from catapult.compiler.base import Compiler
from catapult.runtime.types import dtype


class _NVCCProgram(Compiler):
    def __init__(
        self,
        source: bytes,
        name: bytes,
        kernel_param: bytes,
        device: int,
        compile_options: Optional[List[bytes]] = None,
        num_headers: Optional[int] = None,
        headers: Optional[Tuple[bytes] | List[bytes]] = None,
        include_names: Optional[Tuple[bytes] | List[bytes]] = None,
        template_params: Optional[List[str]] = None,
        nvcc_path: Optional[str] = None,
    ):
        if not isinstance(source, bytes):
            raise CompileException(
                f"Error instantiating NVCC kernel Compiler.",
                f"Value source was passed with ({type(source)}) when it should be of type (bytes).",
            )
        if not isinstance(name, bytes):
            raise CompileException(
                f"Error instantiating NVCC kernel Compiler.",
                f"Value name was passed with ({type(name)}) when it should be of type (bytes).",
            )
        self.source_bytes = source
        self.name_bytes = name
        self.kernel_param = kernel_param or "void*".encode("UTF-8")
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

        self.device = device
        self.compiled_program = None
        self.shared_lib_path = None
        self.mapping = None
        self.named_expression = {}
        self.template_params = template_params
        self.headers = headers
        self.include_names = include_names
        self.num_headers = num_headers

        self.nvcc_path = nvcc_path or self._find_nvcc()

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

        self.compile_options = self._get_options(compile_options)

    def __del__(self):
        # Clean up any temporary files
        if self.shared_lib_path and os.path.exists(self.shared_lib_path):
            try:
                os.remove(self.shared_lib_path)
            except:
                pass

    def _find_nvcc(self) -> str:
        """Find the nvcc executable on the system."""
        try:
            return subprocess.check_output(["which", "nvcc"], text=True).strip()
        except subprocess.CalledProcessError:
            pass

        common_paths = [
            "/usr/local/cuda/bin/nvcc",
            "/usr/local/cuda-11.0/bin/nvcc",
            "/usr/local/cuda-11.1/bin/nvcc",
            "/usr/local/cuda-11.2/bin/nvcc",
            "/usr/local/cuda-11.3/bin/nvcc",
            "/usr/local/cuda-11.4/bin/nvcc",
            "/usr/local/cuda-11.5/bin/nvcc",
            "/usr/local/cuda-11.6/bin/nvcc",
            "/usr/local/cuda-11.7/bin/nvcc",
            "/usr/local/cuda-11.8/bin/nvcc",
            "/usr/local/cuda-12.0/bin/nvcc",
            "/usr/local/cuda-12.1/bin/nvcc",
            "/usr/local/cuda-12.2/bin/nvcc",
            "/usr/local/cuda-12.3/bin/nvcc",
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path

        raise CompileException(
            "Error instantiating NVCC kernel Compiler.",
            "Could not find nvcc executable. Please provide the path to nvcc using the nvcc_path parameter.",
        )

    def get_source(self) -> bytes:
        return self.source_bytes

    def get_name(self) -> bytes:
        return self.name_bytes

    def _create_pybind_module(self, kernel_name: str, kernel_params: str) -> str:
        """
        Create a simple pybind11 module definition for the kernel.

        Args:
            kernel_name: The name of the kernel function
            kernel_params: Parameter types for the kernel

        Returns:
            Pybind11 module code
        """
        return f"""
#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(cuda_example, m) {{ 
    kittens::py::bind_kernel_all<{kernel_name}, {kernel_params}>(m, "{kernel_name}"); 
}}
"""

    def compile(self, template_vals) -> None:
        """
        Compile the CUDA kernel using nvcc.

        Args:
            template_vals: Dictionary of template parameter values
        """
        if not os.path.exists(self.nvcc_path):
            raise CompileException(f"Error compiling kernel: {self.name}", f"NVCC not found at path: {self.nvcc_path}")

        base_temp_dir = os.environ.get("CATAPULT_CACHE_HOME")
        if not base_temp_dir:
            base_temp_dir = os.path.expanduser("~/.cache/catapult")

        os.makedirs(base_temp_dir, exist_ok=True)

        temp_dir = tempfile.mkdtemp(prefix="catapult_nvcc_", dir=base_temp_dir)

        try:
            kernel_name = self.name
            if self.template_params and len(template_vals):
                named_expression, extra_includes = self._create_template_string(template_vals)
                kernel_name = named_expression

            if self.num_headers and self.headers and self.include_names:
                for i in range(self.num_headers):
                    header_content = self.headers[i]
                    include_name = self.include_names[i].decode("UTF-8")
                    header_path = os.path.join(temp_dir, include_name)

                    # Create header file
                    with open(header_path, "wb") as f:
                        f.write(header_content)

            cu_file_path = os.path.join(temp_dir, "kernel.cu")
            with open(cu_file_path, "w") as f:
                f.write(self.source)
                f.write(self._create_pybind_module(kernel_name, self.kernel_param.decode("UTF-8")))
            output_file = os.path.join(temp_dir, "libcuda_example.so")

            arch_flag = f"--gpu-architecture=sm_{self.major}{self.minor}"
            compile_args = [self.nvcc_path]
            for opt in self.compile_options:
                if isinstance(opt, bytes):
                    compile_args.append(opt.decode("ascii"))
                else:
                    compile_args.append(opt)

            compile_args.extend(
                [
                    "--shared",
                    "-Xcompiler",
                    "-fPIC",
                    arch_flag,
                    "-std=c++20",
                    "--expt-relaxed-constexpr",
                    "-fPIC",
                    # Additional optimization and debugging flags
                    "-DNDEBUG",
                    "-Xcompiler=-fPIE",
                    "--expt-extended-lambda",
                    "-Xcompiler=-Wno-psabi",
                    "-Xcompiler=-fno-strict-aliasing",
                    "--use_fast_math",
                    "-forward-unknown-to-host-compiler",
                    "-O3",
                    "-Xnvlink=--verbose",
                    "-Xptxas=--verbose",
                    "-Xptxas=--warn-on-spills",
                    # Additional libraries
                    "-lrt",
                    "-lpthread",
                    "-ldl",
                    "-lcuda",
                    "-lcudadevrt",
                    "-lcudart_static",
                    "-lcublas",
                    "-o",
                    output_file,
                    cu_file_path,
                ]
            )
            process = subprocess.Popen(compile_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            stdout, stderr = process.communicate()

            if process.returncode != 0:
                raise CompileException(
                    f"Error compiling kernel: {self.name}",
                    f"NVCC compilation failed with error:\n{stderr.decode('utf-8')}",
                )

            self.shared_lib_path = output_file

            if self.template_params and len(template_vals):
                self.name = kernel_name
                self.name_bytes = kernel_name.encode("UTF-8")

        except Exception as e:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
            raise e

    def _get_options(self, compile_options):
        """
        Get compilation options for the CUDA kernel, handling GPU architecture specifications.

        Returns:
            List[bytes]: List of compilation options as bytes objects
        """
        options = list(compile_options) if compile_options else []
        options = [opt if isinstance(opt, bytes) else opt.encode("ascii") for opt in options]

        default_opts = [b"--fmad=false", b"-I/home/prairie/Projects/ThunderKittens/include"]

        try:
            import pybind11

            pybind_include = pybind11.get_include()
            default_opts.extend([f"-I{pybind_include}".encode("ascii")])
        except ImportError:
            raise CompileException(
                "Error compiling kernel: {self.name}",
                "Pybind11 include path not found. Ensure pybind11 is installed.",
            )

        try:
            python_includes = subprocess.check_output(["python3-config", "--includes"], text=True).strip()
            for flag in python_includes.split():
                default_opts.append(flag.encode("ascii"))
        except (subprocess.SubprocessError, FileNotFoundError):
            try:
                import sysconfig

                python_include_dir = sysconfig.get_path("include")
                default_opts.append(f"-I{python_include_dir}".encode("ascii"))
            except Exception:
                print("Warning: Could not determine Python include path. Compilation might fail.")
                pass

        try:
            python_ldflags = subprocess.check_output(["python3-config", "--ldflags"], text=True).strip()
            for flag in python_ldflags.split():
                default_opts.append(flag.encode("ascii"))
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        for default_opt in default_opts:
            default_key = default_opt.split(b"=")[0]
            if not any(opt.startswith(default_key) for opt in options):
                options.append(default_opt)

        return options

    def _create_template_string(self, template_vals):
        """
        Create template specialization string for the kernel.

        Args:
            template_vals: Dictionary of template parameter values

        Returns:
            Template string and any extra includes needed
        """
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
        """
        Get the compiled kernel function.

        Returns:
            Tuple of the kernel function and any mapping information
        """
        if self.shared_lib_path is None:
            raise ValueError(
                f"Error accessing kernel '{self.name}': Kernel has not been compiled yet.\n"
                "The compile() method must be called before attempting to get the kernel.\n"
                "This usually happens automatically when the kernel is called with parameters.\n"
                "If you're seeing this error, it may indicate a problem with template parameter resolution "
                "or an issue in the compilation pipeline."
            )

        try:
            import importlib.machinery
            import importlib.util

            loader = importlib.machinery.ExtensionFileLoader("cuda_example", self.shared_lib_path)
            spec = importlib.util.spec_from_loader("cuda_example", loader)
            module = importlib.util.module_from_spec(spec)
            loader.exec_module(module)

            setattr(module, "CATAPULT_NAME", self.name)
            self.compiled_program = module

            return module, self.mapping

        except Exception as e:
            raise CompileException(f"Error loading compiled kernel: {self.name}", f"Failed to load kernel: {str(e)}")
