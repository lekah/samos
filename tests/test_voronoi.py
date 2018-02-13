from samos.analysis.voronoi import VoronoiNetwork
from ase import Atoms
from ase.visualize import view
import numpy as np

NAT = 15
atoms = Atoms('O'*NAT, 10*np.random.random((NAT,3)), cell=(10,10,10))

#~ view(atoms)

vn = VoronoiNetwork()
vn.set_atoms(atoms, 'O')
vn.decompose_qhull()
vn.view_sites()
