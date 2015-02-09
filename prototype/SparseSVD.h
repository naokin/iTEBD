#ifndef __PROTOTYPE_SPARSE_SVD_H
#define __PROTOTYPE_SPARSE_SVD_H

#include <vector>

#include "MPS.h"
#include "Wavefunction.h"

void SparseSVD (
  const std::vector<int>& qR, // Spin quantum #s for row index
  const std::vector<int>& qC, // Spin quantum #s for col index
  const Wavefunction<double>& Wfn,
        std::vector<int>& qS, // Spin quantum #s for selected singular values
        std::vector<std::vector<double>>& lambda,
        MPS<double>& aMps,
        MPS<double>& bMps,
        double CUTOFF_ = 1.0e-16);

#endif // __PROTOTYPE_SPARSE_SVD_H
