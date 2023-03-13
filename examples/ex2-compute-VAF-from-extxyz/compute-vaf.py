from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from samos.analysis.dynamics import DynamicsAnalyzer
from samos.trajectory import Trajectory
from samos.plotting.plot_dynamics import (plot_vaf_isotropic,
                                          plot_power_spectrum)

from ase.io import read


def calculate_plot_vaf(traj):
    da = DynamicsAnalyzer(trajectories=[traj])
    res = da.get_vaf(
        t_end_fit_ps=5,  # where to end fit of MSD
        t_start_fit_ps=1,  # where to start fitting MSD
        nr_of_blocks=4,  # how many independent blocks to calculate
        stepsize_t=1,  # stepsize of times in MSD analysis,
        stepsize_tau=10  # stepsize for time averaging,
    )
    plot_vaf_isotropic(res)
    plt.savefig('vaf.png')


def calculate_plot_vdos(traj):
    da = DynamicsAnalyzer(trajectories=[traj])
    res = da.get_power_spectrum()
    plot_power_spectrum(res)
    plt.savefig('vdos1.png')
    # making my own plot
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(
        GridSpec(1, 1, left=0.15, top=0.95, bottom=0.2, right=0.95)[0])
    for smothening in [1, 10, 100]:
        res = da.get_power_spectrum(smothening=smothening)
        freq_THz = res.get_array('frequency_0')
        vdos = res.get_array('periodogram_Al_mean')
        ax.plot(freq_THz, vdos, label='{}'.format(smothening))
    ax.legend()
    ax.set_yticks([])
    ax.set_ylabel('Signal')
    ax.set_xlabel('Frequency (THz)')
    ax.set_xlim(-2, 22)
    plt.savefig('vdos2.png')
    # ax.plot(res.get_array())


def main():
    filename = '../data/Al31-1200K-2fs.extxyz'
    ase_traj = read(filename, format='extxyz', index=':')
    # sampling was done at every step
    traj = Trajectory.from_atoms(ase_traj, timestep_fs=2.0)

    print("Running Dynamics Analyzer")

    try:
        # this will fail beccause velocities were not set:
        calculate_plot_vaf(traj)
    except Exception as e:
        print(e)

    traj.calculate_velocities_from_positions()

    calculate_plot_vaf(traj)
    calculate_plot_vdos(traj)


if __name__ == '__main__':
    main()
