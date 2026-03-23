import time
from matplotlib import pyplot as plt
from samos.io.lammps import read_lammps_dump
from samos.analysis.dynamics import DynamicsAnalyzer
from samos.plotting.plot_dynamics import plot_msd_isotropic

MSD_KWARGS = dict(
    t_end_fit_ps=100,
    t_start_fit_ps=50,
    species_of_interest=['Li'],
    nr_of_blocks=3,
    stepsize_t=1,
    stepsize_tau=1,
)

N_THREADS_LIST = [1, 2, 4, 8]


def main():
    filename = '../data/LGPS-500K-1ns.lammpstrj'
    print(f"Loading {filename}")
    try:
        traj = read_lammps_dump(filename,)
    except FileNotFoundError:
        print('\nLGPS-500K-1ns.lammpstrj not found. Please uncompress it with:')
        print('tar -xvf LGPS-500K-1ns.lammpstrj.tar.xz')
        return
    traj.set_timestep(1e3)

    da = DynamicsAnalyzer(trajectories=[traj])
    da.set_verbosity(0)
    print("\n--- Fortran implementation ---")
    t0 = time.perf_counter()
    res = da.get_msd(**MSD_KWARGS, backend='fortran')
    t_fortran = time.perf_counter() - t0
    print(f"  {t_fortran:.3f} s")

    print("Making figure")
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    plot_msd_isotropic(res, ax=ax)
    plt.savefig('msd-fortran.png', dpi=150)

    print("\n--- C++ implementation (OpenMP) ---")
    for n in N_THREADS_LIST:
        t0 = time.perf_counter()
        res = da.get_msd(**MSD_KWARGS, backend='cpp', num_threads=n)
        elapsed = time.perf_counter() - t0
        speedup = t_fortran / elapsed
        print(f"  {n:2d} thread(s): {elapsed:.3f} s  (speedup vs Fortran: {speedup:.2f}x)")
        print("Making figure")
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        plot_msd_isotropic(res, ax=ax)
        plt.savefig(f'msd-cpp-{n}.png', dpi=150)


if __name__ == '__main__':
    main()
