from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from samos.io.lammps import read_lammps_dump
from samos.analysis.rdf import RDF
from samos.plotting.plot_rdf import plot_rdf


def main():
    filename = '../data/Al31-1200K-1ps.lammpstrj'
    print(f"Set filename to {filename}")
    traj = read_lammps_dump(filename, elements=['Al']*31)

    print("Running Dynamics Analyzer")
    rdf_analyzer = RDF(trajectory=traj)
    res = rdf_analyzer.run(radius=6)
    print("Making figure 1")
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    plot_rdf(res, ax=ax)
    plt.savefig('rdf-plot1.png', dpi=150)
    print("Arraynames are: " + ', '.join(res.get_arraynames()))

    print("Making figure 2")
    # creating my own plot
    fig = plt.figure(figsize=(8, 3))
    gs = GridSpec(1, 2, left=0.08, bottom=0.15, right=0.98)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    for ax in (ax0, ax1):
        ax.set_xlabel(r'r ($\AA$)')
    ax0.set_ylabel(r'g(r)')
    ax1.set_ylabel(r'integrated number density')
    for stepsize in (1, 10, 100):
        # increasing the sampling stepsize of the RDF
        res = rdf_analyzer.run(radius=6, stepsize=stepsize, istart=1)
        l, = ax0.plot(res.get_array('radii_Al_Al'),
                      res.get_array('rdf_Al_Al'),
                      label=f'stepsize-{stepsize}')
        ax1.plot(res.get_array('radii_Al_Al'),
                 res.get_array('int_Al_Al'), color=l.get_color(),
                 label=f'stepsize-{stepsize}')
    ax0.legend()
    ax1.legend()
    plt.savefig('rdf-plot2.png')
    return


if __name__ == '__main__':
    main()
