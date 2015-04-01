#ifndef __TA_MPS_INIT_H
#define __TA_MPS_INIT_H

#include <vector>

#include "MPS.h"

/// \param qA quantum numbers on the left side of matrix A
/// \param dA size array of each quantum number for qA
/// \param lambdaA diagonal elements of gauge on the left side of matrix A
/// \param mpsA a bundle of A matrices
/// \param qB quantum numbers on the left side of matrix B
/// \param dB size array of each quantum number for qB
/// \param lambdaB diagonal elements of gauge on the left side of matrix B
/// \param mpsB a bundle of B matrices
/// \param M_spin max. spin quantum number (2S)
/// \param M_state number of renormalized states for each symmetry sector (set all symmetries has the same dim.)
void MPS_init (
      madness::World& world,
      std::vector<int>& qA, std::vector<double>& lambdaA, MPS<double>& mpsA,
      std::vector<int>& qB, std::vector<double>& lambdaB, MPS<double>& mpsB,
      int M_spin, size_t M_state);

#endif // __TA_MPS_INIT_H
