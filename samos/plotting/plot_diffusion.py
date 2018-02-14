from matplotlib import pyplot as plt
import numpy as np
from ase.data.colors import jmol_colors
from ase.data import atomic_numbers

def plot_msd_isotropic(msd, ax=None, no_legend=False):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)


        attrs = msd.get_attrs()
        nr_of_blocks = attrs['nr_of_blocks']
        nr_of_trajectories = attrs['nr_of_trajectories']
        block_length_dt = attrs['block_length_dt']
        t_start_fit_dt = attrs['t_start_fit_dt']
        stepsize = attrs.get('stepsize_t', 1)
        timestep_fs = attrs['timestep_fs']

        plt.ylabel(r'MSD $\left[ \AA^2 \right]$')
        plt.xlabel('Time $t$ [fs]')

        times_msd = timestep_fs*stepsize*np.arange(
                    attrs.get('t_start_msd_dt')/stepsize,
                    attrs.get('t_end_msd_dt')/stepsize
                )

        times_fit =  timestep_fs*stepsize*np.arange(
                    attrs.get('t_start_fit_dt')/stepsize,
                    attrs.get('t_end_fit_dt')/stepsize
                )

        for index_of_species, atomic_species in enumerate(attrs['species_of_interest']):
            diff = attrs[atomic_species]['diffusion_mean_cm2_s']
            diff_sem = attrs[atomic_species]['diffusion_sem_cm2_s']
            diff_std = attrs[atomic_species]['diffusion_std_cm2_s']

            #~ axes_msd.plot([],[], , color='w')
            #~ except Exception as e:
                #~ print e
            color = jmol_colors[atomic_numbers[atomic_species]]
            ax.plot([],[], color=color,
                label=r'MSD ({})$\rightarrow D=( {:.2e} \pm {:.2e}) \frac{{cm^2}}{{s}}$'.format(atomic_species, diff, diff_sem))

            for itraj in range(nr_of_trajectories):
                for iblock in range(nr_of_blocks):
                    #~ slope, intercept = 

                #~ lower_bound = timestep_fs * block_index * block_length_dt
                    slope_this_block, intercept_this_block = attrs[atomic_species]['slopes_intercepts'][itraj][iblock]
                    block =  msd.get_array('msd_isotropic_{}_{}_{}'.format(atomic_species, itraj, iblock))

                    ax.plot(
                            times_msd, block,
                            color=color,
                            #~ label=label
                        )
                    ax.plot(times_fit,
                            [slope_this_block*x+intercept_this_block for x in times_fit],
                            color=color, linestyle='--'
                        )
                #~ print block[100],
            #~ print
            #~ try:
                #~ (
                    #~ msd_mean,  msd_sem, msd_std,
                    #~ diff_from_sem_mean, diff_from_sem_err,
                    #~ diff_from_std_mean, diff_from_std_err
                #~ ) = self.msd_averaged[index_of_species]

                #~ (
                    #~ diff_from_sem_mean, diff_from_sem_err,
                    #~ diff_from_std_mean, diff_from_std_err
                #~ ) = [1e4*v for v in  (
                        #~ diff_from_sem_mean, diff_from_sem_err,
                        #~ diff_from_std_mean, diff_from_std_err
                    #~ )] # SI to cm2_s

                #~ plot_msd_averaged = True
            #~ except AttributeError as e:
                # If i was set frmo old results, I don't have this quantity!
                #~ plot_msd_averaged = False

            #~ print msd_mean[100],msd_sem[100],msd_std[100]
            # Filling with SEM


            if 1:
                msd_mean = msd.get_array('msd_isotropic_{}_mean'.format(atomic_species))
                msd_sem = msd.get_array('msd_isotropic_{}_sem'.format(atomic_species))

                p1 = ax.fill_between(
                        times_msd, msd_mean-msd_sem, msd_mean+msd_sem,
                        facecolor='#FFFF00', alpha=0.5, linewidth=1,
                    )
                # Patch to display fill between stuff:
                #http://stackoverflow.com/questions/14534130/legend-not-showing-up-in-matplotlib-stacked-area-plot



        if not(no_legend):
            leg = ax.legend(loc=4)
            leg.get_frame().set_alpha(0.)

    plt.show()
