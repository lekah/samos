import os
import sys
import subprocess
from setuptools import setup
from setuptools.command.build_ext import build_ext

class F2pyBuildExt(build_ext):
    """Custom build command to compile Fortran files using f2py directly."""
    
    def run(self):
        # Ensure numpy is available
        try:
            import numpy
            numpy_include = numpy.get_include()
        except ImportError:
            raise RuntimeError("NumPy must be installed to build extensions.")
        
        # We do NOT call super().run() here because that triggers the standard 
        # Extension compilation logic which fails on .f90 files.
        # Instead, we compile everything manually.
        
        # Define the modules to build
        modules = [
            ('samos.lib.gaussian_density', 'samos/lib/gaussian_density.f90'),
            ('samos.lib.mdutils', 'samos/lib/mdutils.f90'),
            ('samos.lib.rdf', 'samos/lib/rdf.f90'),
        ]
        
        for module_name, source_file in modules:
            self.build_module(module_name, source_file, numpy_include)

    def build_module(self, module_name, source_file, numpy_include):
        """Compile a single Fortran module using f2py."""
        if not os.path.exists(source_file):
            print(f"Warning: Source file {source_file} not found, skipping.")
            return

        source_dir = os.path.dirname(os.path.abspath(source_file))
        module_short_name = module_name.split('.')[-1]
        
        # Construct the f2py command
        # We use the system f2py module
        cmd = [
            sys.executable, "-m", "numpy.f2py",
            "-c", source_file,
            "-m", module_short_name,
            f"--include-paths={numpy_include}",
            "--f90flags=-fPIC"
        ]
        
        # Change to the source directory to run f2py
        old_cwd = os.getcwd()
        try:
            os.chdir(source_dir)
            print(f"Compiling {source_file} -> {module_short_name}...")
            subprocess.check_call(cmd)
            
            # f2py creates the .so file in the current directory (source_dir)
            # We need to move it to the correct package location if building inplace
            # or let setuptools handle the installation later.
            # For 'build_ext --inplace', the .so ends up in source_dir, which is correct
            # if source_dir is part of the package (samos/lib/).
            
        except subprocess.CalledProcessError as e:
            print(f"Error compiling {source_file}: {e}")
            raise
        finally:
            os.chdir(old_cwd)

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
        'packages': ['samos', 'samos.lib'],
        'package_data': {'samos.lib': ['*.f90']},
        # IMPORTANT: Do NOT include ext_modules here. 
        # The custom build command handles everything.
        'cmdclass': {'build_ext': F2pyBuildExt},
    }

    setup(**setup_kwargs)