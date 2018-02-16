from matplotlib import pyplot as plt
import numpy as np

from ase.data import atomic_numbers
from samos.utils.colors import get_color



def plot_msd_isotropic(msd, 
        ax=None, no_legend=False, species_of_interest=None, show=False, **kwargs):
    if ax is None:
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(1,1,1)
    attrs = msd.get_attrs()
    if attrs['decomposed']:
        raise NotImplementedError("Plotting decomposed trajectories is not implemented")

    nr_of_trajectories = attrs['nr_of_trajectories']
    t_start_fit_dt = attrs['t_start_fit_dt']
    stepsize = attrs.get('stepsize_t', 1)
    timestep_fs = attrs['timestep_fs']

    ax.set_ylabel(r'MSD $\left[ \AA^2 \right]$')
    ax.set_xlabel('Time $t$ [fs]')

    times_msd = timestep_fs*stepsize*np.arange(
                attrs.get('t_start_dt')/stepsize,
                attrs.get('t_end_dt')/stepsize
            )

    times_fit =  timestep_fs*stepsize*np.arange(
                attrs.get('t_start_fit_dt')/stepsize,
                attrs.get('t_end_fit_dt')/stepsize
            )
    if species_of_interest is None:
        species_of_interest = attrs['species_of_interest']
    for index_of_species, atomic_species in enumerate(species_of_interest):
        diff = attrs[atomic_species]['diffusion_mean_cm2_s']
        diff_sem = attrs[atomic_species]['diffusion_sem_cm2_s']
        diff_std = attrs[atomic_species]['diffusion_std_cm2_s']
        color = get_color(atomic_species, scheme='cpk')
        msd_mean = msd.get_array('msd_isotropic_{}_mean'.format(atomic_species))
        msd_sem = msd.get_array('msd_isotropic_{}_sem'.format(atomic_species))
        p1 = ax.fill_between(
                times_msd, msd_mean-msd_sem, msd_mean+msd_sem,
                facecolor=color, alpha=.2, linewidth=1,
            )
        ax.plot(times_msd,msd_mean, color=color, linewidth=3.,
            label=r'MSD ({})$\rightarrow D=( {:.2e} \pm {:.2e}) \frac{{cm^2}}{{s}}$'.format(atomic_species, diff, diff_sem))

        for itraj in range(nr_of_trajectories):
            msd_this_traj =  msd.get_array('msd_isotropic_{}_{}'.format(atomic_species, itraj))
            slopes_intercepts_this_traj =  msd.get_array('slopes_intercepts_isotropic_{}_{}'.format(atomic_species, itraj))
            for iblock in range(len(msd_this_traj)):
                slope_this_block, intercept_this_block = slopes_intercepts_this_traj[iblock]
                ax.plot(times_msd, msd_this_traj[iblock], color=color, alpha=0.1,)
                ax.plot(times_fit, [slope_this_block*x+intercept_this_block for x in times_fit], color=color, linestyle='--', alpha=0.2)
    if not(no_legend):
        leg = ax.legend(loc=4)
        leg.get_frame().set_alpha(0.)
    if show:
        plt.show()
def plot_vaf_isotropic(vaf, 
        ax=None, no_legend=False, species_of_interest=None, show=False, **kwargs):
    from matplotlib.ticker import ScalarFormatter
    f = ScalarFormatter()
    f.set_powerlimits((-1,1))

    if ax is None:
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(1,1,1)
    attrs = vaf.get_attrs()

    nr_of_trajectories = attrs['nr_of_trajectories']
    t_start_fit_dt = attrs['t_start_fit_dt']
    t_end_fit_dt = attrs['t_end_fit_dt']
    stepsize = attrs.get('stepsize_t', 1)
    timestep_fs = attrs['timestep_fs']

    ax.set_ylabel(r'MSD $\left[ \AA^2 \right]$')
    ax.set_xlabel('Time $t$ [fs]')
    axes_D = ax.twinx()
    axes_D.yaxis.set_major_formatter(f)
    axes_D.set_ylabel(r"$\int^t_0 VAF(t') dt' \quad \left[ \frac{cm^2}{s} \right]$")
    maxy = 0 # to set reasonable ylimits for axes_D, I track the max diff by hand

    times = timestep_fs*stepsize*np.arange(
                attrs.get('t_start_dt')/stepsize,
                attrs.get('t_end_dt')/stepsize
            )

    times_fit =  timestep_fs*stepsize*np.arange(
                attrs.get('t_start_fit_dt')/stepsize,
                attrs.get('t_end_fit_dt')/stepsize
            )
    if species_of_interest is None:
        species_of_interest = attrs['species_of_interest']

    for index_of_species, atomic_species in enumerate(species_of_interest):
        diff = attrs[atomic_species]['diffusion_mean_cm2_s']
        diff_sem = attrs[atomic_species]['diffusion_sem_cm2_s']
        diff_std = attrs[atomic_species]['diffusion_std_cm2_s']
        color = get_color(atomic_species, scheme='cpk')
        vaf_mean = vaf.get_array('vaf_isotropic_{}_mean'.format(atomic_species))
        vaf_sem = vaf.get_array('vaf_isotropic_{}_sem'.format(atomic_species))
        vaf_integral_mean = vaf.get_array('vaf_integral_isotropic_{}_mean'.format(atomic_species))
        vaf_integral_sem = vaf.get_array('vaf_integral_isotropic_{}_sem'.format(atomic_species))


        ax.fill_between(
                times, vaf_mean-vaf_sem, vaf_mean+vaf_sem,
                facecolor=color, alpha=.2, linewidth=1,
            )
        ax.plot(times,vaf_mean, color=color, linewidth=3.,
            label=r'VAF ({})'.format(atomic_species))


        maxy = max((maxy, (vaf_integral_mean+vaf_integral_sem).max()))
        for itraj in range(nr_of_trajectories):
            vaf_this_traj =  vaf.get_array('vaf_isotropic_{}_{}'.format(atomic_species, itraj))
            vaf_integral_this_traj =  vaf.get_array('vaf_integral_isotropic_{}_{}'.format(atomic_species, itraj))
            maxy = max((maxy, vaf_integral_this_traj.max()))

            for iblock in range(len(vaf_this_traj)):
                ax.plot(times, vaf_this_traj[iblock], color=color, alpha=0.1,)
                axes_D.plot(times, vaf_integral_this_traj[iblock], color=color, alpha=0.1, linestyle='--',)

        axes_D.plot(times ,vaf_integral_mean, color=color, linewidth=3., linestyle='--',
                label=r'D ({})$\rightarrow D=( {:.2e} \pm {:.2e}) \frac{{cm^2}}{{s}}$'.format(atomic_species, diff, diff_sem))
        axes_D.fill_between(times, vaf_integral_mean-vaf_integral_sem, vaf_integral_mean+vaf_integral_sem, 
                facecolor=color, alpha=.2, linewidth=1)


    axes_D.set_ylim(0,maxy)
    axes_D.axvline(t_start_fit_dt*timestep_fs*stepsize, color='grey', linewidth=2, alpha=0.2,)
    axes_D.axvline(t_end_fit_dt*timestep_fs*stepsize, color='grey', linewidth=2, alpha=0.2)

    if not(no_legend):
        leg = ax.legend(loc=4)
        leg.get_frame().set_alpha(0.)
        leg = axes_D.legend(loc=1)
        leg.get_frame().set_alpha(0.)
    if show:
        plt.show()
