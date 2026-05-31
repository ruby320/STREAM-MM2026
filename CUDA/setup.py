"""Build timetest-local CUDA extension (stquant_timetest_cpp). Run: pip install -e ."""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="stquant_timetest",
    ext_modules=[
        CUDAExtension(
            name="stquant_timetest_cpp",
            sources=[
                "stquant_core.cpp",
                "stquant_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
