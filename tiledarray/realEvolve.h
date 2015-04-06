#ifndef __TA_REAL_EVOLVE_H
#define __TA_REAL_EVOLVE_H

#include <vector>

#include "MPS.h"

/// Make a Trotter step of real time-evolution
/// DEF.: A = mpsA, p = lambdaA, B = mpsB, q = lambdaB
/// ALGO:
/// 1) A = q.a, B = p.b
/// 2) C = A.B.q = q.a.p.b.q = A'.p.B' by doing SVD
/// 3) normalize(p)
/// 4) A"= A'= q.a
/// 5) B"= p.B'.(1/q) = p.b
double realEvolve (
      madness::World& world,
      std::vector<int>& qA, std::vector<double>& lambdaA, MPS<double>& mpsA_real, MPS<double>& mpsA_imag,
      std::vector<int>& qB, std::vector<double>& lambdaB, MPS<double>& mpsB_real, MPS<double>& mpsB_imag,
      double J, double Jz, double Hz, double dt, double tole);

#endif // __TA_REAL_EVOLVE_H
