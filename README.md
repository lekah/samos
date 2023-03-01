# SAMOS (Suite for Analysis of Molecular Simulations)

This software, distributed openly and free of charge (see LICENCE), analyzes molecular dynamics simulations.
Currently implemented are:

  * The estimate of tracer and charge diffusion coefficients from the mean-square displacements or the integral of the velocity-autocorrelation function
  * The calculation of atomic probability densities
  * Radial distribution functions
  * Several plotting utilities

This code is written mostly in Python 3, the computationally intensive functions are in fortran90 and wrapped with f2py.
To install, clone or download this repository, move into the directory of this file, and type:

    pip install .

It will install the package and dependencies.
