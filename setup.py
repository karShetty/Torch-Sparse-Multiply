from __future__ import annotations

import os

import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CUDAExtension


def get_extensions():
    include_dirs = ['spspmm']
    main_source = os.path.join('spspmm', 'sparse_matmul_extension.cpp')
    source_cuda = [os.path.join('spspmm', 'sparse.cu')]
    sources = [main_source]

    extension = CppExtension

    extra_compile_args = {'cxx': ['-std=c++17']}
    define_macros = []

    force_cuda = os.getenv('FORCE_CUDA', '0') == '1'
    if (torch.cuda.is_available() and CUDA_HOME is not None) or force_cuda:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [('WITH_CUDA', None)]
        nvcc_args = [
            '-DCUDA_HAS_FP16=1',
            '-extended-lambda',
        ]
        nvcc_flags_env = os.getenv('NVCC_FLAGS', '')
        if nvcc_flags_env != '':
            nvcc_args.extend(nvcc_flags_env.split(' '))

        CC = os.environ.get('CC', None)
        if CC is not None:
            CC_arg = f'-ccbin={CC}'
            if CC_arg not in nvcc_args:
                if any(arg.startswith('-ccbin') for arg in nvcc_args):
                    raise ValueError('Inconsistent ccbins')
                nvcc_args.append(CC_arg)

        extra_compile_args['nvcc'] = nvcc_args

    ext_modules = [
        extension(
            'spspmm._C',
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            libraries=['cusparse'],
        ),
    ]

    return ext_modules


if os.getenv('NO_NINJA', '0') == '1':

    class BuildExtension(torch.utils.cpp_extension.BuildExtension):
        def __init__(self, *args, **kwargs):
            super().__init__(use_ninja=False, *args, **kwargs)


else:
    BuildExtension = torch.utils.cpp_extension.BuildExtension

package_name = 'spspmm'
long_description = 'A pytorch module to compute sparse sparse matrix multiplication.'

setup(
    name='spspmm',
    description='Pytorch Sparse Sparse Matrix Multiplication',
    packages=find_packages(),
    install_requires=[],
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension},
)
