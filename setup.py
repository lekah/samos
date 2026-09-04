# -*- coding: utf-8 -*-
# Metadata is declared in pyproject.toml.
# This file handles only the Fortran (f2py) extension build.

import glob
import os
import shutil
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class f2py_Extension(Extension):

    def __init__(self, name, sourcedirs):
        Extension.__init__(self, name, sources=[])
        self.sourcedirs = [os.path.abspath(s) for s in sourcedirs]
        self.dirs = sourcedirs


class F2PyBuild(build_ext):
    """Builds the f2py (Fortran) extensions; nothing else runs here."""

    def run(self):
        use_stdlib_distutils = sys.version_info < (3, 12)

        for ext in self.extensions:
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


setup(
    # No compiled extensions are built any more (issues.md issues #1,
    # #2, #9) -- mdutils.f90, mdutils_cpp_omp.cpp and
    # gaussian_density.f90 are all kept in samos/lib for reference,
    # each with a header explaining what replaced it, but none is
    # listed here. f2py_Extension and F2PyBuild stay in case compiled
    # code comes back.
    ext_modules=[],
    cmdclass=dict(build_ext=F2PyBuild),
)
