import os
import sys
import glob
import shutil
import subprocess
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyWithFortran(build_py):
    """Extend build_py to also compile Fortran extensions via f2py."""

    def run(self):
        # First do the normal Python build
        super().run()

        # numpy is guaranteed by pyproject.toml build requirements
        modules = [
            ('samos.lib.gaussian_density', 'samos/lib/gaussian_density.f90'),
            ('samos.lib.mdutils',          'samos/lib/mdutils.f90'),
            ('samos.lib.rdf',              'samos/lib/rdf.f90'),
        ]

        for module_name, source_file in modules:
            self._build_f2py_module(module_name, source_file)

    def _build_f2py_module(self, module_name, source_file):
        if not os.path.exists(source_file):
            print(f"Warning: {source_file} not found, skipping.")
            return

        source_dir = os.path.dirname(os.path.abspath(source_file))
        module_short_name = module_name.split('.')[-1]
        package_path = os.path.join(*module_name.split('.')[:-1])  # e.g. samos/lib

        cmd = [
            sys.executable, "-m", "numpy.f2py",
            "--backend", "meson",
            "-c", os.path.basename(source_file),
            "-m", module_short_name,
        ]

        old_cwd = os.getcwd()
        try:
            os.chdir(source_dir)
            print(f"Compiling {source_file} -> {module_short_name}...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd)
        except subprocess.CalledProcessError as e:
            print(f"Error compiling {source_file}: {e}")
            raise
        finally:
            os.chdir(old_cwd)

        # Copy the compiled .so into the build lib directory so it gets installed
        pattern = os.path.join(source_dir, f"{module_short_name}*.so")
        so_files = glob.glob(pattern)
        if not so_files:
            # Also check for .pyd on Windows
            pattern = os.path.join(source_dir, f"{module_short_name}*.pyd")
            so_files = glob.glob(pattern)

        dest_dir = os.path.join(self.build_lib, package_path)
        os.makedirs(dest_dir, exist_ok=True)
        for so in so_files:
            dest = os.path.join(dest_dir, os.path.basename(so))
            print(f"Copying {so} -> {dest}")
            shutil.copy2(so, dest)


if __name__ == '__main__':
    setup_kwargs = {
        'author': 'Leonid Kahle',
        'author_email': 'leonid.kahle@epfl.ch',
        'classifiers': [
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Operating System :: OS Independent',
            'Topic :: Scientific/Engineering :: Physics',
            'Topic :: Scientific/Engineering :: Chemistry',
            'Topic :: Software Development :: Libraries :: Python Modules'
        ],
        'description': 'Package for Analysis and Tricks for MOlecular Simulations',
        'extras_require': {
            'pre-commit': ['pre-commit', 'flake8'],
            'docs': ['Sphinx', 'docutils', 'sphinx_rtd_theme']
        },
        'install_requires': [
            'numpy>=1.14.0',
            'ase>=3.17.0',
            'scipy>=1.0.0',
            'matplotlib>=2.1.2'
        ],
        'license': 'MIT License',
        'name': 'samos',
        'url': 'https://github.com/lekah/samos',
        'version': '0.8',
        'packages': ['samos', 'samos.lib', 'samos.utils', 'samos.io', 'samos.analysis', 'samos.plotting'],
        'package_data': {'samos.lib': ['*.f90']},
        'cmdclass': {'build_py': BuildPyWithFortran},
    }

    setup(**setup_kwargs)
