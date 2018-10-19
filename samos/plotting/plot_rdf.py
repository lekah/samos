from matplotlib import pyplot as plt
import numpy as np

from ase.data import atomic_numbers
from samos.utils.colors import get_color
from copy import deepcopy

def plot_rdf(rdf_res,
        ax=None, ax2=None, no_legend=False, species_of_interest=None, show=False, label=None, no_label=False,
        alpha_fill=0.2, alpha_block=0.3, alpha_fit=0.4, color_scheme='jmol', exclude_from_label=None, plot_params={}, plot_params2={}):


    if ax is None:
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(1,1,1)
        

    if ax2 is None:
        ax2 = ax.twinx()

    attrs = rdf_res.get_attrs()

    handles = []
    for spec1, spec2 in attrs['species_pairs']:
        rdf = rdf_res.get_array('rdf_{}_{}'.format(spec1, spec2))
        integral = rdf_res.get_array('int_{}_{}'.format(spec1, spec2))
        radii = rdf_res.get_array('radii_{}_{}'.format(spec1, spec2))
        plot_params_ = deepcopy(plot_params)
        plot_params2_ = deepcopy(plot_params2)
        
        if 'color' in plot_params_:
            pass
        elif 'colordict' in plot_params_:
            plot_params_['color'] = plot_params_.pop('colordict')['{}_{}'.format(spec1, spec2)]
        if 'label' not in plot_params_ and not no_label:
            if 'labelspec' in plot_params_:
                labelspec = plot_params_.pop('labelspec')
                plot_params_['label'] = r'$g(r)_{{{}-{}}}$ {}'.format(spec1, spec2, labelspec)
            else:
                plot_params_['label'] = r'$g(r)$ {}-{}'.format(spec1, spec2)
        if 'label' not in plot_params2_ and not no_label:
            plot_params2_['label'] = r'$\int g(r)$ {} {}'.format(spec1, spec2)
        l, = ax.plot(radii, rdf, **plot_params_)
        handles.append(l)
        if 'color' in plot_params2_:
            pass
        if 'colordict'in plot_params2_:
            plot_params2_['color'] = plot_params2_.pop('colordict')['{}_{}'.format(spec1, spec2)]
        else:
            plot_params2_['color'] = l.get_color()
        l2, = ax2.plot(radii, integral, '--', **plot_params2_)
        handles.append(l2)

    ax.set_xlabel(r'r / $\AA$')
    ax.set_ylabel(r'$g(r)$')
    ax2.set_ylabel(r'$\int g(r) dr$')
    if show:
        plt.show()
    return handles
        

def plot_angular_spec(angspec_res,
        ax=None, no_legend=False, species_of_interest=None, show=False, label=None, no_label=False,
        alpha_fill=0.2, alpha_block=0.3, alpha_fit=0.4, color_scheme='jmol', exclude_from_label=None, **kwargs):


    if ax is None:
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(1,1,1)
    attrs = angspec_res.get_attrs()
    handles = []
    for spec1, spec2, spec3 in attrs['species_pairs']:
        angular_spec = angspec_res.get_array('aspec_{}_{}_{}'.format(spec1, spec2, spec3))
        angles = angspec_res.get_array('angles_{}_{}_{}'.format(spec1, spec2, spec3))
        if not no_label:
            label1 = r'$g(r)$ {}-{}-{}'.format(spec2, spec1, spec3)
        else:
            label1 = None
        l, = ax.plot(angles, angular_spec, label=label1)
        handles.append(l)
    plt.legend(handles=handles)
    #~ ax.set_xlabel(r'r / $\AA$')
    #~ ax.set_ylabel(r'$g(r)$')
    #~ ax2.set_ylabel(r'$\int g(r) dr$')
    if show:
        plt.show()
        
