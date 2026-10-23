"""Build the optional native extensions.

* ``kiln._C``      — C++ (always buildable): block radix tree for the router.
* ``kiln._cuda``   — CUDA (built when nvcc is found): fused log-prob/entropy, RMSNorm.

``pip install -e .`` without nvcc builds only ``_C``; the Triton / PyTorch
paths cover the GPU kernels in that case. Set ``KILN_SKIP_NATIVE=1`` to build
neither (pure-Python install, e.g. CI lint jobs).
"""

from __future__ import annotations

import os

from setuptools import setup

ext_modules = []
cmdclass = {}
if os.environ.get("KILN_SKIP_NATIVE", "0") != "1":
    try:
        from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CppExtension, CUDAExtension

        ext_modules.append(
            CppExtension("kiln._C", ["csrc/radix_tree.cpp"], extra_compile_args=["-O3"])
        )
        if CUDA_HOME is not None and os.environ.get("KILN_SKIP_CUDA", "0") != "1":
            ext_modules.append(
                CUDAExtension(
                    "kiln._cuda",
                    ["csrc/bindings.cpp", "csrc/fused_logprob_entropy.cu", "csrc/rmsnorm.cu"],
                    include_dirs=[os.path.abspath("csrc")],
                    extra_compile_args={
                        "cxx": ["-O3"],
                        "nvcc": ["-O3", "--use_fast_math", "-lineinfo",
                                 "-gencode=arch=compute_80,code=sm_80",
                                 "-gencode=arch=compute_90,code=sm_90",
                                 "-gencode=arch=compute_90,code=compute_90"],
                    },
                )
            )
        cmdclass["build_ext"] = BuildExtension.with_options(use_ninja=False)
    except ImportError as e:  # torch missing from the build env
        raise SystemExit(
            "kiln native build needs torch in the build environment: "
            "pip install torch && pip install --no-build-isolation -e . "
            "(or KILN_SKIP_NATIVE=1 for a pure-Python install)"
        ) from e

setup(ext_modules=ext_modules, cmdclass=cmdclass)
