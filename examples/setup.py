from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Get CUDA compute capability
device = torch.cuda.get_device_properties(0)
arch = f"{device.major}{device.minor}"

setup(
    name='saxpy_cuda',
    version='0.1',
    author='User',
    author_email='user@example.com',
    description='SAXPY CUDA Extension',
    ext_modules=[
        CUDAExtension('saxpy_cuda', 
            sources=['saxpy_torch.cu'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-lineinfo',
                    f'-arch=sm_{arch}',
                    '-std=c++17',
                    '--extended-lambda',
                    '--expt-relaxed-constexpr'
                ]
            },
            include_dirs=[
                '/usr/local/cuda/include',
                '/usr/local/cuda-12/include',  # Add CUDA 12 specific path
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.7.0',
    ]
)