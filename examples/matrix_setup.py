from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

device = torch.cuda.get_device_properties(0)
arch = f"{device.major}{device.minor}"

setup(
    name='matrix_cuda',
    version='0.1',
    author='User',
    author_email='user@example.com',
    description='Complex Matrix Operations CUDA Extension',
    ext_modules=[
        CUDAExtension('matrix_cuda', 
            sources=['matrix_torch.cu'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '--use_fast_math',
                    '-lineinfo',
                    f'-arch=sm_{arch}',
                    '-std=c++17',
                ]
            },
            include_dirs=[
                '/usr/local/cuda/include',
                '/usr/local/cuda-12/include',
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