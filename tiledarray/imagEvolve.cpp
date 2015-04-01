#include "imagEvolve.h"

#include <iostream>
#include <iomanip>

#include "Wavefunction.h"
#include "GetWavefunction.h"
#include "F_gauge_fix.h"
#include "TA_sparse_svd.h"

/// Make a Trotter step of imaginary time-evolution for the ground state search
/// DEF.: A = mpsA, p = lambdaA, B = mpsB, q = lambdaB
/// ALGO:
/// 1) A = q.a, B = p.b
/// 2) C = A.B.q = q.a.p.b.q = A'.p.B' by doing SVD
/// 3) normalize(p)
/// 4) A"= A'= q.a
/// 5) B"= p.B'.(1/q) = p.b
double imagEvolve (
      madness::World& world,
      std::vector<int>& qA, std::vector<double>& lambdaA, MPS<double>& mpsA,
      std::vector<int>& qB, std::vector<double>& lambdaB, MPS<double>& mpsB,
      double J, double Jz, double Hz, double dt, double tole)
{
  r_gauge_fix(lambdaB,mpsB);

  Wavefunction<double> wfn; GetWavefunction(mpsA,mpsB,wfn);

  double wfnNorm2 = wfn.norm2(world);

  // Compute exp(-h*dt)*wfn

  // Nearest neighbour propagator
  //         +1                                                     -1
  //         +1                         -1                          +1                          -1
  // ------+--------------------------------------------------------------------------------------------------------------
  // +1 +1 |  exp(-Jz*dt/4)*exp(-Hz*dt)  0                           0                           0
  //    -1 |  0                          exp(+Jz*dt/4)*cosh(J*dt/2) -exp(+Jz*dt/4)*sinh(J*dt/2)  0
  // -1 +1 |  0                         -exp(+Jz*dt/4)*sinh(J*dt/2)  exp(+Jz*dt/4)*cosh(J*dt/2)  0
  //    -1 |  0                          0                           0                           exp(-Jz*dt/4)*exp(+Hz*dt)

  double expJz = exp(-0.25*Jz*dt);
  double expHz = exp(-Hz*dt);
  double coshJ = cosh(0.5*J*dt);
  double sinhJ = sinh(0.5*J*dt);

  {
    auto ud_tmp = wfn.matrix_ud;
    auto du_tmp = wfn.matrix_du;

    // NOTE: there are no operator implementations *=, /=, +=, -= in TiledArray class (?)

    for(auto it = wfn.matrix_uu.begin(); it != wfn.matrix_uu.end(); ++it) it->get() *= (expJz*expHz);

    wfn.matrix_ud("i,j") = (coshJ/expJz)*ud_tmp("i,j")-(sinhJ/expJz)*du_tmp("i,j");

    wfn.matrix_du("i,j") = (coshJ/expJz)*du_tmp("i,j")-(sinhJ/expJz)*ud_tmp("i,j");

    for(auto it = wfn.matrix_dd.begin(); it != wfn.matrix_dd.end(); ++it) it->get() *= (expJz/expHz);
  }

  double sgvNorm2 = wfn.norm2(world);

  TA_sparse_svd(world,qB,qB,wfn,qA,lambdaA,mpsA,mpsB,tole);

  double aNorm2 = 0.0;
  for(size_t k = 0; k < lambdaA.size(); ++k) aNorm2 += lambdaA[k]*lambdaA[k];

  double aNorm  = sqrt(aNorm2);
  for(size_t k = 0; k < lambdaA.size(); ++k) lambdaA[k] /= aNorm;

  l_gauge_fix        (lambdaA,mpsB);

  r_gauge_fix_inverse(lambdaB,mpsB);

  return -log(sgvNorm2)/wfnNorm2/dt/2.0;
}
