#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> calculate_msd_specific_atoms(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions,
    py::array_t<int, py::array::c_style | py::array::forcecast> indices,
    int stepsize_t, int stepsize_tau, int block_length_dt, int nr_of_blocks,
    int nr_of_t, int nstep, int nat, int nat_of_interest)
{
    auto pos = positions.unchecked<3>();
    auto idx = indices.unchecked<1>();

    py::array_t<double> msd({nr_of_blocks, nr_of_t});
    auto m = msd.mutable_unchecked<2>();

    for (int iblock = 0; iblock < nr_of_blocks; iblock++) {
        for (int t = 1; t <= nr_of_t; t++) {
            double acc = 0.0;
            for (int ai = 0; ai < nat_of_interest; ai++) {
                int iat = idx(ai) - 1;  // convert 1-based to 0-based
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
            m(iblock, t - 1) = acc;
        }
    }
    double norm = (double)block_length_dt / (double)stepsize_tau
                * (double)nat_of_interest;
    for (int iblock = 0; iblock < nr_of_blocks; iblock++)
        for (int t = 0; t < nr_of_t; t++)
            m(iblock, t) /= norm;

    return msd;
}

py::array_t<double> calculate_msd_specific_atoms_decompose_d(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions,
    py::array_t<int, py::array::c_style | py::array::forcecast> indices,
    int stepsize, int stepsize_inner, int block_length_dt, int nr_of_blocks,
    int nr_of_t, int nstep, int nat, int nat_of_interest)
{
    auto pos = positions.unchecked<3>();
    auto idx = indices.unchecked<1>();

    py::array_t<double> msd({nr_of_blocks, nr_of_t, 3, 3});
    auto m = msd.mutable_unchecked<4>();

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
                    m(iblock, t - 1, ipol, jpol) = acc;
                }
            }
        }
    }
    double norm = (double)block_length_dt / (double)stepsize_inner
                * (double)nat_of_interest;
    for (int iblock = 0; iblock < nr_of_blocks; iblock++)
        for (int t = 0; t < nr_of_t; t++)
            for (int ipol = 0; ipol < 3; ipol++)
                for (int jpol = 0; jpol < 3; jpol++)
                    m(iblock, t, ipol, jpol) /= norm;

    return msd;
}

py::array_t<double> calculate_msd_specific_atoms_max_stats(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions,
    py::array_t<int, py::array::c_style | py::array::forcecast> indices,
    int stepsize_t, int /*stepsize_tau*/, int nr_of_t, int nstep, int nat,
    int nat_of_interest)
    // stepsize_tau is not used but the fortran part uses it so it is left here to indicate that
{
    auto pos = positions.unchecked<3>();
    auto idx = indices.unchecked<1>();

    py::array_t<double> msd(nr_of_t);
    auto m = msd.mutable_unchecked<1>();

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
        m(t - 1) = running_mean;
    }
    return msd;
}

PYBIND11_MODULE(msd_utils, m) {
    m.def("calculate_msd_specific_atoms", &calculate_msd_specific_atoms);
    m.def("calculate_msd_specific_atoms_decompose_d",
          &calculate_msd_specific_atoms_decompose_d);
    m.def("calculate_msd_specific_atoms_max_stats",
          &calculate_msd_specific_atoms_max_stats);
}
