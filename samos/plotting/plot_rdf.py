from matplotlib import pyplot as plt
import numpy as np

from ase.data import atomic_numbers
from samos.utils.colors import get_color


def plot_rdf(rdf_res,
        ax=None, no_legend=False, species_of_interest=None, show=False, label=None, no_label=False,
        alpha_fill=0.2, alpha_block=0.3, alpha_fit=0.4, color_scheme='jmol', exclude_from_label=None, **kwargs):


    if ax is None:
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(1,1,1)
    attrs = rdf_res.get_attrs()
    ax2 = ax.twinx()
    handles = []
    for spec1, spec2 in attrs['species_pairs']:
        rdf = rdf_res.get_array('rdf_{}_{}'.format(spec1, spec2))
        integral = rdf_res.get_array('int_{}_{}'.format(spec1, spec2))
        radii = rdf_res.get_array('radii_{}_{}'.format(spec1, spec2))
        if not no_label:
            label1 = r'$g(r)$ {} {}'.format(spec1, spec2)
            label2 = r'$\int g(r)$ {} {}'.format(spec1, spec2)
            #~ label2 = None
        else:
            label2 = None
        l, = ax.plot(radii, rdf, label=label1)
        handles.append(l)
        l2, = ax2.plot(radii, integral, '--', color=l.get_color(), label=label2)
        #~ handles.append(l2)
    plt.legend(handles=handles)
    ax.set_xlabel(r'r / $\AA$')
    ax.set_ylabel(r'$g(r)$')
    ax2.set_ylabel(r'$\int g(r) dr$')
    if show:
        plt.show()
        
