#include "imagEvolve.h"

#include <iostream>
#include <iomanip>

#include "Wavefunction.h"
#include "SqNorm.h"
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

  world.gop.fence(); // FIXME: does this need?

  Wavefunction<double> wfn;

  wfn.matrix_uu("i,j") = mpsA.matrix_u("i,k")*mpsB.matrix_u("k,j");

  wfn.matrix_ud("i,j") = mpsA.matrix_u("i,k")*mpsB.matrix_d("k,j");

  wfn.matrix_du("i,j") = mpsA.matrix_d("i,k")*mpsB.matrix_u("k,j");

  wfn.matrix_dd("i,j") = mpsA.matrix_d("i,k")*mpsB.matrix_d("k,j");

  world.gop.fence(); // FIXME: does this need?

  double wfnNorm2 = SqNorm(world,wfn);

  // Compute exp(-h*dt)*wfn

  // Nearest neighbour propagator
  //         +1                                                     -1
  //         +1                         -1                          +1                          -1
  // ------+--------------------------------------------------------------------------------------------------------------
  // +1 +1 |  exp(-Jz*dt/4)*exp(+Hz*dt)  0                           0                           0
  //    -1 |  0                          exp(+Jz*dt/4)*cosh(J*dt/2)  exp(+Jz*dt/4)*sinh(J*dt/2)  0
  // -1 +1 |  0                          exp(+Jz*dt/4)*sinh(J*dt/2)  exp(+Jz*dt/4)*cosh(J*dt/2)  0
  //    -1 |  0                          0                           0                           exp(-Jz*dt/4)*exp(-Hz*dt)

  Wavefunction<double> sgv;

  double expJz = exp(-0.25*Jz*dt);
  double expHz = exp(Hz*dt);
  double coshJ = cosh(0.5*J*dt);
  double sinhJ = sinh(0.5*J*dt);

  sgv.matrix_uu("i,j") = (expJz*expHz)*wfn.matrix_uu("i,j");

  sgv.matrix_ud("i,j") = (coshJ/expJz)*wfn.matrix_ud("i,j")+(sinhJ/expJz)*wfn.matrix_du("i,j");

  sgv.matrix_du("i,j") = (coshJ/expJz)*wfn.matrix_du("i,j")+(sinhJ/expJz)*wfn.matrix_ud("i,j");

  sgv.matrix_dd("i,j") = (expJz/expHz)*wfn.matrix_dd("i,j");

  world.gop.fence(); // FIXME: does this need?

  double sgvNorm2 = SqNorm(world,sgv);

  TA_sparse_svd(world,qB,qB,sgv,qA,lambdaA,mpsA,mpsB,tole);

  world.gop.fence(); // FIXME: does this need?

  double aNorm2 = 0.0;
  for(size_t k = 0; k < lambdaA.size(); ++k) aNorm2 += lambdaA[k]*lambdaA[k];

  double aNorm  = sqrt(aNorm2);
  for(size_t k = 0; k < lambdaA.size(); ++k) lambdaA[k] /= aNorm;

  l_gauge_fix        (lambdaA,mpsB);

  world.gop.fence(); // FIXME: does this need?

  r_gauge_fix_inverse(lambdaB,mpsB);

  world.gop.fence(); // FIXME: does this need?

  return -log(sgvNorm2)/wfnNorm2/dt/2.0;
}
