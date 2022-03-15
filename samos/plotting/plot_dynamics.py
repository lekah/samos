# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np

from ase.data import atomic_numbers
from samos.utils.colors import get_color

def my_format(real, decimals=2):
    exp = np.floor(np.log10(real))
    pref = real / 10**exp
    return '{:.{prec}f} \cdot 10^{{{}}}'.format(pref, int(exp), prec=decimals)

def format_mean_err(mean, err, decimals=2):
    if np.isnan(mean):
        return 'N/A'
    mean_rounded_to_prec = float('{:.{prec}e}'.format(mean, prec=decimals))
    exp_mean = int(np.floor(np.log10(np.abs(mean_rounded_to_prec))))
    pref_mean = mean_rounded_to_prec / 10.0**exp_mean
    if np.isnan(err):
        return '{:.{prec}f}\cdot 10^{{{}}}'.format(pref_mean, exp_mean, prec=decimals)
    else:
        err_rounded_to_prec = float('{:.{prec}e}'.format(err, prec=decimals))
        exp_err = int(np.floor(np.log10(np.abs(err_rounded_to_prec))))
        pref_err = err_rounded_to_prec / 10.0**exp_err
        if exp_mean == exp_err:
            return '\left({:.{prec}f} \pm {:.{prec}f} \\right)\cdot 10^{{{}}}'.format(pref_mean, pref_err, exp_mean, prec=decimals)
        else:
            return '{:.{prec}f} \cdot 10^{{{}}} \pm {:.{prec}f} \cdot 10^{{{}}}'.format(
                pref_mean,exp_mean, pref_err, exp_err, prec=decimals)



def plot_msd_isotropic(msd,
        ax=None, no_legend=False, species_of_interest=None, show=False, label=None, no_label=False,
        alpha_fill=0.2, alpha_block=0.3, alpha_fit=0.4, color_scheme='jmol', exclude_from_label=None,
        color_dict={}, decimals=1, no_block_fits=False, no_long=False, grid=False, **kwargs):

    if ax is None:
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(1,1,1)
    attrs = msd.get_attrs()
    if attrs['decomposed']:
        raise NotImplementedError('Plotting decomposed trajectories is not implemented')
    multiple_params_fit = attrs.get('multiple_params_fit', False)

    nr_of_trajectories = attrs['nr_of_trajectories']
    t_start_fit_dt = attrs['t_start_fit_dt']
    stepsize = attrs.get('stepsize_t', 1)
    timestep_fs = attrs['timestep_fs']
    plot_long = attrs.get('do_long', False) and not(no_long)

    ax.set_ylabel(r'$\mathrm{MSD}(t)$ $\left( \mathrm{\AA}^2 \right) $ ')
    ax.set_xlabel(r'$t$ $\left( \mathrm{ps}\right)$')

    times_msd = msd.get_array('t_list_fs') / 1e3

    if not no_block_fits:
        times_fit =  timestep_fs / 1000.0 * stepsize * np.arange(
            attrs.get('t_start_fit_dt') // stepsize,
            attrs.get('t_end_fit_dt') // stepsize
        )
    if species_of_interest is None:
        species_of_interest = attrs['species_of_interest']
    for index_of_species, atomic_species in enumerate(species_of_interest):
        diff = attrs[atomic_species]['diffusion_mean_cm2_s']
        diff_sem = attrs[atomic_species]['diffusion_sem_cm2_s']
        diff_std = attrs[atomic_species]['diffusion_std_cm2_s']
        if atomic_species in color_dict:
            color = color_dict[atomic_species]
        else:
            color = get_color(atomic_species, scheme=color_scheme)
        msd_mean = msd.get_array('msd_isotropic_{}_mean'.format(atomic_species))
        msd_sem = msd.get_array('msd_isotropic_{}_sem'.format(atomic_species))
        p1 = ax.fill_between(
                times_msd, msd_mean-msd_sem, msd_mean+msd_sem,
                facecolor=color, alpha=alpha_fill, linewidth=1,
            )
        if no_label or (exclude_from_label and atomic_species in exclude_from_label):
            label_this_species = None
        elif label is None:
            if not multiple_params_fit:
                label_this_species = r'$D_{{\mathrm{{{}}}}}={} \, \frac{{cm^2}}{{s}}$'.format(
                    atomic_species, format_mean_err(diff, diff_sem, decimals=decimals))
            else:
                label_this_species = r'{}'.format(atomic_species)
        else:
            label_this_species = '{}'.format(label)

        if plot_long:
            # reduce number of lines in plot, customize for later!
            # Keep the legend though!
            ax.plot([],[], color=color, linewidth=1.0, label=label_this_species)
            ax.plot(times_msd, msd_mean, color=color, linewidth=2.)
        else:
            ax.plot(times_msd, msd_mean, color=color, linewidth=2., label=label_this_species)


        for itraj in range(nr_of_trajectories):
            msd_this_traj =  msd.get_array('msd_isotropic_{}_{}'.format(atomic_species, itraj))
            slopes_intercepts_this_traj =  msd.get_array('slopes_intercepts_isotropic_{}_{}'.format(atomic_species, itraj))
            for iblock in range(len(msd_this_traj)):
                ax.plot(times_msd, msd_this_traj[iblock], color=color, alpha=alpha_block, lw=0.5)
                if not no_block_fits:
                    slope_this_block, intercept_this_block = slopes_intercepts_this_traj[iblock]
                    ax.plot(times_fit, [1000.*slope_this_block*x+intercept_this_block for x in times_fit], color=color, linestyle='--', lw=1.0, alpha=alpha_fit)
            if plot_long:
                times_long = msd.get_array('t_list_long_fs')[itraj] / 1e3
                ax.plot(times_long, msd.get_array('msd_long_{}_{}'.format(atomic_species, itraj)), color=color, linestyle='-', lw=1.0)
    if not(no_legend):
        leg = ax.legend(loc=2,labelspacing=0.01)
        leg.get_frame().set_alpha(0.)
    if grid:
        ax.grid(ls=':')
    if show:
        plt.show()
    return ax

def plot_msd_anisotropic(msd,
        ax=None, no_legend=False, species_of_interest=None, show=False, label=None, no_label=False,
        alpha_fill=0.2, alpha_block=0.3, alpha_fit=0.4, color_scheme='jmol', exclude_from_label=None,
        diagonal_only=False, label_diagonal=True, no_block_fits=False, grid=False, **kwargs):

    if ax is None:
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(1,1,1)
    attrs = msd.get_attrs()
    if not(attrs['decomposed']):
        raise NotImplementedError('Only plotting decomposed with this functions')
    multiple_params_fit = attrs.get('multiple_params_fit', False)

    nr_of_trajectories = attrs['nr_of_trajectories']
    t_start_fit_dt = attrs['t_start_fit_dt']
    stepsize = attrs.get('stepsize_t', 1)
    timestep_fs = attrs['timestep_fs']

    ax.set_ylabel(r'$\mathrm{MSD}(t)$ $\left( \mathrm{\AA}^2 \right) $ ')
    ax.set_xlabel(r'$t$ $\left( \mathrm{ps}\right)$')

    times_msd = msd.get_array('t_list_fs') / 1e3

    if not no_block_fits:
        times_fit =  timestep_fs / 1000.0 * stepsize * np.arange(
            attrs.get('t_start_fit_dt') // stepsize,
            attrs.get('t_end_fit_dt') // stepsize
        )
    if species_of_interest is None:
        species_of_interest = attrs['species_of_interest']

    colors = ['r', 'orange', 'orange', 'orange', 'g', 'orange','orange', 'orange', 'b']
    for index_of_species, atomic_species in enumerate(species_of_interest):
        diff = attrs[atomic_species]['diffusion_mean_cm2_s']
        diff_sem = attrs[atomic_species]['diffusion_sem_cm2_s']
        diff_std = attrs[atomic_species]['diffusion_std_cm2_s']
        #color = get_color(atomic_species, scheme=color_scheme)
        msd_mean = msd.get_array('msd_decomposed_{}_mean'.format(atomic_species))
        msd_sem = msd.get_array('msd_decomposed_{}_sem'.format(atomic_species))

        if no_label or (exclude_from_label and atomic_species in exclude_from_label):
            label_this_species = False
        else:
            label_this_species = True
        count=0
        for i in range(3):
            for j in range(3):
                color = colors[count]
                count += 1
                if ( diagonal_only and i!=j):
                    continue
                if label_this_species and (i==j or label_diagonal):
                    if multiple_params_fit:
                        label = r'$\mathrm{{{}}}_{{{}{}}}$'.format(atomic_species, i, j)
                    else:
                        label = r'$D_{{{}{}}}^{{{}}}=( {:.1e} \pm {:.1e}) \frac{{cm^2}}{{s}}$'.format(
                                    i, j, atomic_species, diff[i][j], diff_sem[i][j])
                else:
                    label = None

                ax.plot(times_msd,msd_mean[:,i,j], color=color,
                        linewidth=2., label=label)
                ax.fill_between(times_msd, msd_mean[:,i,j] - msd_sem[:,i,j], msd_mean[:,i,j] + msd_sem[:,i,j],
                                                            facecolor=color, alpha=alpha_fill, linewidth=1)
                for itraj in range(nr_of_trajectories):
                    msd_this_traj =  msd.get_array('msd_decomposed_{}_{}'.format(atomic_species, itraj))
                    slopes_intercepts_this_traj =  msd.get_array('slopes_intercepts_decomposed_{}_{}'.format(atomic_species, itraj))
                    for iblock in range(len(msd_this_traj)):
                        ax.plot(times_msd, msd_this_traj[iblock,:,i,j], color=color, alpha=alpha_block,lw=0.5, zorder=1)
                        if not no_block_fits:
                            slope_this_block, intercept_this_block = slopes_intercepts_this_traj[iblock][i][j]
                            ax.plot(times_fit, [1000.*slope_this_block*x+intercept_this_block for x in times_fit],
                                    color=color, linestyle='--', alpha=alpha_fit, zorder=2, lw=1.0)
    if not(no_legend):
        leg = ax.legend(loc=2)
        leg.get_frame().set_alpha(0.)
    if grid:
        ax.grid(ls=':')
    if show:
        plt.show()
    return ax

def plot_vaf_isotropic(vaf,
        ax=None, no_legend=False, species_of_interest=None, show=False,
        color_scheme='jmol', **kwargs):
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

    ax.set_ylabel(r'VAF $\left[ \AA^2 fs^{-2} \right]$')
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
        color = get_color(atomic_species, scheme=color_scheme)
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
                label=r'$D_{{{}}}^{{VAF}}=( {:.2e} \pm {:.2e}) \frac{{cm^2}}{{s}}$'.format(atomic_species, diff, diff_sem))
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


def plot_power_spectrum(power_spectrum, ax=None, show=False, color_scheme='jmol', alpha_signals=0.1,
                        alpha_fill=0.2, **kwargs):
    if ax is None:
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(1,1,1)
    attrs = power_spectrum.get_attrs()
    ax.set_xlabel(r'$\omega$ $\left[THz\right]$')
    ax.set_ylabel(r'Signal $[\AA^2 fs^{-1}]$')
    species_of_interest = attrs['species_of_interest']
    nr_of_trajectories = attrs['nr_of_trajectories']
    frequencies = [power_spectrum.get_array('frequency_{}'.format(itraj)) for itraj in range(nr_of_trajectories)]
    for index_of_species, atomic_species in enumerate(species_of_interest):
        color = get_color(atomic_species, scheme=color_scheme)
        if alpha_signals > 1e-4:
            for itraj in range(nr_of_trajectories):
                freq = frequencies[itraj]
                periodogram = power_spectrum.get_array('periodogram_{}_{}'.format( atomic_species, itraj))
                for signal in periodogram:
                    ax.plot(freq, signal,color=color, alpha=alpha_signals)
        try:
            periodogram_mean = power_spectrum.get_array('periodogram_{}_mean'.format( atomic_species))
            periodogram_sem = power_spectrum.get_array('periodogram_{}_sem'.format( atomic_species))
            ax.plot(frequencies[0], periodogram_mean, color=color, alpha=1, linewidth=1)
            ax.fill_between(frequencies[0], periodogram_mean-periodogram_sem, periodogram_mean+periodogram_sem,
                facecolor=color, alpha=alpha_fill, linewidth=1)
        except Exception as e:
            print(e)
    if show:
        plt.show()
