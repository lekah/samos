#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <vector>

// Only contains calculate_msd functions along with get_com_positions for ease of import when switching between fortran and C++
// All functions are taken as is from the fortran code to preserve the logic

namespace py = pybind11;

void set_num_threads(int n) { omp_set_num_threads(n); }

py::array_t<double> get_com_positions(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions,
    py::array_t<double, py::array::c_style | py::array::forcecast> masses,
    py::array_t<int,    py::array::c_style | py::array::forcecast> factors)
{
    auto pos = positions.unchecked<3>();
    auto m_  = masses.unchecked<1>();
    auto fac = factors.unchecked<1>();

    int nstep = (int)positions.shape(0);
    int nat   = (int)positions.shape(1);

    std::vector<double> rel_masses(nat);
    double M = 0.0;
    for (int i = 0; i < nat; i++) { rel_masses[i] = fac(i) * m_(i); M += rel_masses[i]; }
    for (int i = 0; i < nat; i++) rel_masses[i] /= M;

    py::array_t<double> out({nstep, 1, 3});
    auto o = out.mutable_unchecked<3>();

    #pragma omp parallel for schedule(static)
    for (int istep = 0; istep < nstep; istep++) {
        double cx = 0.0, cy = 0.0, cz = 0.0;
        for (int iat = 0; iat < nat; iat++) {
            cx += rel_masses[iat] * pos(istep, iat, 0);
            cy += rel_masses[iat] * pos(istep, iat, 1);
            cz += rel_masses[iat] * pos(istep, iat, 2);
        }
        o(istep, 0, 0) = cx;
        o(istep, 0, 1) = cy;
        o(istep, 0, 2) = cz;
    }
    return out;
}

py::array_t<double> calculate_msd_specific_atoms(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions,
    py::array_t<int,    py::array::c_style | py::array::forcecast> indices,
    int stepsize_t, int stepsize_tau, int block_length_dt, int nr_of_blocks,
    int nr_of_t, int nstep, int nat, int nat_of_interest)
{
    auto pos = positions.unchecked<3>();
    auto idx = indices.unchecked<1>();

    py::array_t<double> msd({nr_of_blocks, nr_of_t});
    auto mv = msd.mutable_unchecked<2>();

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int iblock = 0; iblock < nr_of_blocks; iblock++) {
        for (int t = 1; t <= nr_of_t; t++) {
            double acc = 0.0;
            for (int ai = 0; ai < nat_of_interest; ai++) {
                int iat = idx(ai) - 1;  // convert 1 based to 0 base index
                for (int tau = iblock * block_length_dt;
                     tau < (iblock + 1) * block_length_dt;
                     tau += stepsize_tau) {
                    for (int ipol = 0; ipol < 3; ipol++) {
                        double d = pos(tau + stepsize_t * t, iat, ipol)
                                 - pos(tau, iat, ipol);
                        acc += d * d;
                    }
                }
            }
            mv(iblock, t - 1) = acc;
        }
    }
    // Matches Fortran: msd / block_length_dt / nat_of_interest * stepsize_tau
    double norm = (double)block_length_dt / (double)stepsize_tau * (double)nat_of_interest;
    for (int iblock = 0; iblock < nr_of_blocks; iblock++)
        for (int t = 0; t < nr_of_t; t++)
            mv(iblock, t) /= norm;

    return msd;
}

py::array_t<double> calculate_msd_specific_atoms_decompose_d(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions,
    py::array_t<int,    py::array::c_style | py::array::forcecast> indices,
    int stepsize, int stepsize_inner, int block_length_dt, int nr_of_blocks,
    int nr_of_t, int nstep, int nat, int nat_of_interest)
{
    auto pos = positions.unchecked<3>();
    auto idx = indices.unchecked<1>();

    py::array_t<double> msd({nr_of_blocks, nr_of_t, 3, 3});
    auto mv = msd.mutable_unchecked<4>();

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int iblock = 0; iblock < nr_of_blocks; iblock++) {
        for (int t = 1; t <= nr_of_t; t++) {
            for (int ipol = 0; ipol < 3; ipol++) {
                for (int jpol = 0; jpol < 3; jpol++) {
                    double acc = 0.0;
                    for (int ai = 0; ai < nat_of_interest; ai++) {
                        int iat = idx(ai) - 1;
                        for (int tau = iblock * block_length_dt;
                             tau < (iblock + 1) * block_length_dt;
                             tau += stepsize_inner) {
                            acc += (pos(tau + stepsize * t, iat, ipol)
                                  - pos(tau, iat, ipol))
                                 * (pos(tau + stepsize * t, iat, jpol)
                                  - pos(tau, iat, jpol));
                        }
                    }
                    mv(iblock, t - 1, ipol, jpol) = acc;
                }
            }
        }
    }
    // Matches Fortran: msd / block_length_dt / nat_of_interest * stepsize_inner
    double norm = (double)block_length_dt / (double)stepsize_inner * (double)nat_of_interest;
    for (int iblock = 0; iblock < nr_of_blocks; iblock++)
        for (int t = 0; t < nr_of_t; t++)
            for (int ipol = 0; ipol < 3; ipol++)
                for (int jpol = 0; jpol < 3; jpol++)
                    mv(iblock, t, ipol, jpol) /= norm;

    return msd;
}

// stepsize_tau is not used (iterates every step), but is accepted for compatibility with python call
py::array_t<double> calculate_msd_specific_atoms_max_stats(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions,
    py::array_t<int,    py::array::c_style | py::array::forcecast> indices,
    int stepsize_t, int /*stepsize_tau*/, int nr_of_t, int nstep, int nat,
    int nat_of_interest)
{
    auto pos = positions.unchecked<3>();
    auto idx = indices.unchecked<1>();

    py::array_t<double> msd(nr_of_t);
    auto mv = msd.mutable_unchecked<1>();

    #pragma omp parallel for schedule(dynamic)
    for (int t = 1; t <= nr_of_t; t++) {
        double running_mean = 0.0;
        int icount = 1;
        for (int ai = 0; ai < nat_of_interest; ai++) {
            int iat = idx(ai) - 1;
            for (int tau = 0; tau < nstep - t * stepsize_t; tau++) {
                double disp2 = 0.0;
                for (int ipol = 0; ipol < 3; ipol++) {
                    double d = pos(tau + stepsize_t * t, iat, ipol)
                             - pos(tau, iat, ipol);
                    disp2 += d * d;
                }
                double fc = (double)icount;
                running_mean = (fc - 1.0) / fc * running_mean + disp2 / fc;
                icount++;
            }
        }
        mv(t - 1) = running_mean;
    }
    return msd;
}

PYBIND11_MODULE(mdutils_cpp_omp, m) {
    m.def("set_num_threads", &set_num_threads, "Set number of OpenMP threads");
    m.def("get_com_positions", &get_com_positions);
    m.def("calculate_msd_specific_atoms", &calculate_msd_specific_atoms);
    m.def("calculate_msd_specific_atoms_decompose_d",
          &calculate_msd_specific_atoms_decompose_d);
    m.def("calculate_msd_specific_atoms_max_stats",
          &calculate_msd_specific_atoms_max_stats);
}
