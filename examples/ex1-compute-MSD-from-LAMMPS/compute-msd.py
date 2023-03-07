from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from samos.io.lammps import read_lammps_dump
from samos.analysis.dynamics import DynamicsAnalyzer
from samos.plotting.plot_dynamics import plot_msd_isotropic

def main():
    filename = '../data/Al31-1200K-1ps.lammpstrj'
    print(f"Set filename to {filename}")
    traj = read_lammps_dump(filename, elements=['Al']*31)
    traj.set_timestep(1e3) # sampling was done every 1000fs = 1ps

    print("Running Dynamics Analyzer")
    da = DynamicsAnalyzer(trajectories=[traj])
    res = da.get_msd(t_end_fit_ps=100, # where to end fit of MSD
                     t_start_fit_ps=50, # where to start fitting MSD
                     species_of_interest=['Al',],  # can be left empty
                     # all will be calculated by default
                     nr_of_blocks=4, # how many independent blocks to calculate
                     stepsize_t=1, # stepsize of times in MSD analysis,
                     stepsize_tau=10 # stepsize for time averaging,
                )
    print("Making figure 1")
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    plot_msd_isotropic(res,ax=ax )
    plt.savefig('msd-plot1.png', dpi=150)

    # creating my own plot
    print("Making figure 2")
    print("Arraynames are: " + ', '.join(res.get_arraynames()))
    print("attributes calculated for Al are:")
    for key, val in res.get_attrs()["Al"].items():
        print('   {:<21}: {}'.format(key, val))
    fig = plt.figure(figsize=(8,3))
    gs = GridSpec(1,2, left=0.08, bottom=0.15, right=0.98)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    for ax in (ax0, ax1):
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel(r'MSD $(\AA^2)$')
    ax0.set_title('Mean MSD')
    ax1.set_title('Block MSDs')
    times_ps = res.get_array('t_list_fs') / 1e3
    ax0.plot(times_ps, res.get_array('msd_isotropic_Al_mean'))
    for iblock, block in enumerate(res.get_array('msd_isotropic_Al_0')):
        ax1.plot(times_ps, block, label=f'block-{iblock}')
    ax1.legend()
    
    plt.savefig('msd-plot2.png')



if __name__ == '__main__':
    main()