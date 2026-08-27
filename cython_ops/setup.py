"""Standalone build of the tenxnet Ridgepath Cython ops (feasibility check).

Builds the preprocessing (target generation) and postprocessing (instance decode) +
metrics Cython modules OUTSIDE bazel, so they can be reused from a PyTorch project
without rewriting the algorithms. The .pyx are copied verbatim from tenxnet (unmodified).

    python setup.py build_ext --inplace
"""
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

NP = np.get_include()
OMP = ["-fopenmp"]  # ridgepath_construct uses cython.parallel.prange

extensions = [
    # preprocessing: dense ridgepath target generation (uses prange -> OpenMP)
    Extension(
        "ridgepath_construct",
        ["ridgepath_construct.pyx"],
        include_dirs=[NP, "."],
        extra_compile_args=["-O2"] + OMP,
        extra_link_args=OMP,
    ),
    # postprocessing: instance decode stack
    Extension("centerpath_post_cy", ["centerpath_post_cy.pyx"], include_dirs=[NP, "."], extra_compile_args=["-O2"]),
    Extension(
        "centerpath_construction",
        ["centerpath_construction.pyx"],
        include_dirs=[NP, "."],
        extra_compile_args=["-O2"] + OMP,
        extra_link_args=OMP,
    ),
    Extension("follow_flows", ["follow_flows.pyx"], include_dirs=[NP, "."], extra_compile_args=["-O2"]),
    Extension("heap", ["heap.pyx"], include_dirs=[NP, "."], extra_compile_args=["-O2"]),
    # label_morph uses libcpp -> C++
    Extension(
        "label_morph",
        ["label_morph.pyx"],
        language="c++",
        include_dirs=[NP, "."],
        extra_compile_args=["-O2", "-std=c++17"],
    ),
    # metrics: COCO-style mask API (companion C source)
    Extension("_mask", ["_mask.pyx", "maskApi.c"], include_dirs=[NP, "."], extra_compile_args=["-O2"]),
]

setup(
    name="ridgepath_cython_ops",
    ext_modules=cythonize(
        extensions,
        language_level=3,
        compiler_directives={"profile": False},  # override the .pyx profile=True
    ),
)
