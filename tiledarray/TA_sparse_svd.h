#ifndef __TA_SPARSE_SVD_H
#define __TA_SPARSE_SVD_H

#include <vector>

#include "MPS.h"
#include "Wavefunction.h"

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

#endif // __TA_SPARSE_SVD_H
