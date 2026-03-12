# -*- coding: utf-8 -*-
# Metadata is declared in pyproject.toml.
# This file handles only the C++ (pybind11) and Fortran (f2py) extension builds.

import glob
import os
import shutil
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from pybind11.setup_helpers import Pybind11Extension


class f2py_Extension(Extension):

    def __init__(self, name, sourcedirs):
        Extension.__init__(self, name, sources=[])
        self.sourcedirs = [os.path.abspath(s) for s in sourcedirs]
        self.dirs = sourcedirs


class CombinedBuild(build_ext):
    """Builds f2py (Fortran) and pybind11 (C++) extensions in one pass."""

    def run(self):
        f2py_exts = [e for e in self.extensions if isinstance(e, f2py_Extension)]
        cpp_exts  = [e for e in self.extensions if not isinstance(e, f2py_Extension)]

        # Standard setuptools path initialises the compiler; run for C++ only.
        self.extensions = cpp_exts
        build_ext.run(self)

        use_stdlib_distutils = sys.version_info < (3, 12)

        for ext in f2py_exts:
            for i, src in enumerate(ext.sourcedirs):
                module_loc = os.path.split(ext.dirs[i])[0]
                module_name = os.path.split(src)[1].split('.')[0]
                env = os.environ.copy()
                if use_stdlib_distutils:
                    env['SETUPTOOLS_USE_DISTUTILS'] = 'stdlib'
                subprocess.check_call(
                    [sys.executable, '-m', 'numpy.f2py', '-c', src, '-m', module_name],
                    cwd=module_loc, env=env
                )
                # Copy the built .so into the build tree so setuptools includes
                # it in non-editable installs (regular pip install).
                dest_dir = os.path.join(self.build_lib, module_loc)
                os.makedirs(dest_dir, exist_ok=True)
                for so in glob.glob(os.path.join(module_loc, module_name + '*.so')):
                    shutil.copy(so, dest_dir)

        self.extensions = cpp_exts + f2py_exts


setup(
    ext_modules=[
        f2py_Extension('fortran_lib', [
            'samos/lib/gaussian_density.f90',
            'samos/lib/mdutils.f90',
            'samos/lib/rdf.f90',
        ]),
        Pybind11Extension(
            'samos.lib.mdutils_cpp_omp',
            ['samos/lib/mdutils_cpp_omp.cpp'],
            extra_compile_args=['-O3', '-fopenmp'],
            extra_link_args=['-fopenmp'],
        ),
    ],
    cmdclass=dict(build_ext=CombinedBuild),
)
