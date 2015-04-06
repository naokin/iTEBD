#ifndef __TA_SPARSE_SVD_H
#define __TA_SPARSE_SVD_H

#include <vector>

#include "MPS.h"
#include "Wavefunction.h"

/// perform SVD on wavefunction
void TA_sparse_svd (
      madness::World& world,
const std::vector<int>& qR, /// Spin quantum #s for row index (replicated)
const std::vector<int>& qC, /// Spin quantum #s for col index (replicated)
const Wavefunction<double>& wfn,
      std::vector<int>& qS, /// Spin quantum #s for selected singular values
      std::vector<double>& lambda,
      MPS<double>& mpsA,
      MPS<double>& mpsB,
      double CUTOFF_ = 1.0e-16);

/// perform SVD on complex wavefunction
/// real and imaginary parts are stored separately
void TA_complex_sparse_svd (
      madness::World& world,
const std::vector<int>& qR, /// Spin quantum #s for row index (replicated)
const std::vector<int>& qC, /// Spin quantum #s for col index (replicated)
const Wavefunction<double>& wfn_real,
const Wavefunction<double>& wfn_imag,
      std::vector<int>& qS, /// Spin quantum #s for selected singular values
      std::vector<double>& lambda,
      MPS<double>& mpsA_real,
      MPS<double>& mpsA_imag,
      MPS<double>& mpsB_real,
      MPS<double>& mpsB_imag,
      double CUTOFF_ = 1.0e-16);

#endif // __TA_SPARSE_SVD_H
