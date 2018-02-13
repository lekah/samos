import numpy as np

from samos.trajectory import Trajectory
from samos.analysis.get_gaussian_density import get_gaussian_density
t = Trajectory(positions=np.load('test.npy'), cell=np.diag([10,10,10]), symbols=['H'])


get_gaussian_density(t) 
