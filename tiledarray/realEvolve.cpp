#include "realEvolve.h"

#include <iostream>
#include <iomanip>

#include "Wavefunction.h"
#include "SqNorm.h"
#include "F_gauge_fix.h"
#include "TA_sparse_svd.h"

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
      double J, double Jz, double Hz, double dt, double tole)
{
//std::cout << "DEBUG[" << world.rank() << "] : 00" << std::endl;
  r_gauge_fix(lambdaB,mpsB_real);
  r_gauge_fix(lambdaB,mpsB_imag);

  world.gop.fence(); // FIXME: does this need?
//std::cout << "DEBUG[" << world.rank() << "] : 01" << std::endl;

  Wavefunction<double> wfn_real;

  wfn_real.matrix_uu("i,j") = mpsA_real.matrix_u("i,k")*mpsB_real.matrix_u("k,j")-mpsA_imag.matrix_u("i,k")*mpsB_imag.matrix_u("k,j");

  wfn_real.matrix_ud("i,j") = mpsA_real.matrix_u("i,k")*mpsB_real.matrix_d("k,j")-mpsA_imag.matrix_u("i,k")*mpsB_imag.matrix_d("k,j");

  wfn_real.matrix_du("i,j") = mpsA_real.matrix_d("i,k")*mpsB_real.matrix_u("k,j")-mpsA_imag.matrix_d("i,k")*mpsB_imag.matrix_u("k,j");

  wfn_real.matrix_dd("i,j") = mpsA_real.matrix_d("i,k")*mpsB_real.matrix_d("k,j")-mpsA_imag.matrix_d("i,k")*mpsB_imag.matrix_d("k,j");

  world.gop.fence();
//std::cout << "DEBUG[" << world.rank() << "] : 02" << std::endl;

  Wavefunction<double> wfn_imag;

  wfn_imag.matrix_uu("i,j") = mpsA_imag.matrix_u("i,k")*mpsB_real.matrix_u("k,j")+mpsA_real.matrix_u("i,k")*mpsB_imag.matrix_u("k,j");

  wfn_imag.matrix_ud("i,j") = mpsA_imag.matrix_u("i,k")*mpsB_real.matrix_d("k,j")+mpsA_real.matrix_u("i,k")*mpsB_imag.matrix_d("k,j");

  wfn_imag.matrix_du("i,j") = mpsA_imag.matrix_d("i,k")*mpsB_real.matrix_u("k,j")+mpsA_real.matrix_d("i,k")*mpsB_imag.matrix_u("k,j");

  wfn_imag.matrix_dd("i,j") = mpsA_imag.matrix_d("i,k")*mpsB_real.matrix_d("k,j")+mpsA_real.matrix_d("i,k")*mpsB_imag.matrix_d("k,j");

  world.gop.fence();
//std::cout << "DEBUG[" << world.rank() << "] : 03" << std::endl;

  double wfnNorm2 = SqNorm(world,wfn_real)+SqNorm(world,wfn_imag);

  // Compute exp(-i*h*dt)*wfn

  double cosJz = cos(-0.25*Jz*dt);
  double sinJz = sin(-0.25*Jz*dt);
  double cosHz = cos(Hz*dt);
  double sinHz = sin(Hz*dt);
  double cosJ  = cos(0.5*J*dt);
  double sinJ  = cos(0.5*J*dt);

  // Nearest neighbour propagator (real part)
  //         +1                                                     -1
  //         +1                         -1                          +1                          -1
  // ------+--------------------------------------------------------------------------------------------------------------
  // +1 +1 |  cos(-Jz*dt/4+Hz*dt)        0                           0                           0
  //    -1 |  0                          cos(+Jz*dt/4)*cos(J*dt/2)   sin(-Jz*dt/4)*sin(J*dt/2)   0
  // -1 +1 |  0                          sin(-Jz*dt/4)*sin(J*dt/2)   cos(+Jz*dt/4)*cos(J*dt/2)   0
  //    -1 |  0                          0                           0                           cos(-Jz*dt/4-Hz*dt)

  // Nearest neighbour propagator (imag part)
  //         +1                                                     -1
  //         +1                         -1                          +1                          -1
  // ------+--------------------------------------------------------------------------------------------------------------
  // +1 +1 |  sin(-Jz*dt/4+Hz*dt)        0                           0                           0
  //    -1 |  0                          sin(+Jz*dt/4)*cos(J*dt/2)   cos(+Jz*dt/4)*sin(J*dt/2)   0
  // -1 +1 |  0                          cos(+Jz*dt/4)*sin(J*dt/2)   sin(+Jz*dt/4)*cos(J*dt/2)   0
  //    -1 |  0                          0                           0                           sin(-Jz*dt/4-Hz*dt)

  // propagated wave (real part) = Re[exp(-i*h*dt)]*Re[wfn]-Im[exp(-i*h*dt)]*Im[wfn]

  Wavefunction<double> sgv_real;

  sgv_real.matrix_uu("i,j") = (cosJz*cosHz-sinJz*sinHz)*wfn_real.matrix_uu("i,j")-(sinJz*cosHz+cosJz*sinHz)*wfn_imag.matrix_uu("i,j");

  sgv_real.matrix_ud("i,j") = (cosJz*cosJ)*wfn_real.matrix_ud("i,j")+(sinJz*sinJ)*wfn_real.matrix_du("i,j")+(sinJz*cosJ)*wfn_imag.matrix_ud("i,j")-(cosJz*sinJ)*wfn_imag.matrix_du("i,j");

  sgv_real.matrix_du("i,j") = (sinJz*sinJ)*wfn_real.matrix_ud("i,j")+(cosJz*cosJ)*wfn_real.matrix_du("i,j")-(cosJz*sinJ)*wfn_imag.matrix_ud("i,j")+(sinJz*cosJ)*wfn_imag.matrix_du("i,j");

  sgv_real.matrix_dd("i,j") = (cosJz*cosHz+sinJz*sinHz)*wfn_real.matrix_dd("i,j")-(sinJz*cosHz-cosJz*sinHz)*wfn_imag.matrix_dd("i,j");

  world.gop.fence();
//std::cout << "DEBUG[" << world.rank() << "] : 04" << std::endl;

  // propagated wave (real part) = Im[exp(-i*h*dt)]*Re[wfn]+Im[exp(-i*h*dt)]*Im[wfn]

  Wavefunction<double> sgv_imag;

  sgv_imag.matrix_uu("i,j") = (sinJz*cosHz+cosJz*sinHz)*wfn_real.matrix_uu("i,j")+(cosJz*cosHz-sinJz*sinHz)*wfn_imag.matrix_uu("i,j");

  sgv_imag.matrix_ud("i,j") =-(sinJz*cosJ)*wfn_real.matrix_ud("i,j")+(cosJz*sinJ)*wfn_real.matrix_du("i,j")+(cosJz*cosJ)*wfn_imag.matrix_ud("i,j")+(sinJz*sinJ)*wfn_imag.matrix_du("i,j");

  sgv_imag.matrix_du("i,j") = (cosJz*sinJ)*wfn_real.matrix_ud("i,j")-(sinJz*cosJ)*wfn_real.matrix_du("i,j")+(sinJz*sinJ)*wfn_imag.matrix_ud("i,j")+(cosJz*cosJ)*wfn_imag.matrix_du("i,j");

  sgv_imag.matrix_dd("i,j") = (sinJz*cosHz-cosJz*sinHz)*wfn_real.matrix_dd("i,j")+(cosJz*cosHz+sinJz*sinHz)*wfn_imag.matrix_dd("i,j");

  world.gop.fence();
//std::cout << "DEBUG[" << world.rank() << "] : 05" << std::endl;

  double sgvNorm2 = SqNorm(world,sgv_real)+SqNorm(world,sgv_imag);

  TA_complex_sparse_svd(world,qB,qB,sgv_real,sgv_imag,qA,lambdaA,mpsA_real,mpsA_imag,mpsB_real,mpsB_imag,tole);

  world.gop.fence(); // FIXME: does this need?
//std::cout << "DEBUG[" << world.rank() << "] : 06" << std::endl;

  double aNorm2 = 0.0;
  for(size_t k = 0; k < lambdaA.size(); ++k) aNorm2 += lambdaA[k]*lambdaA[k];

  double aNorm  = sqrt(aNorm2);
  for(size_t k = 0; k < lambdaA.size(); ++k) lambdaA[k] /= aNorm;

  l_gauge_fix        (lambdaA,mpsB_real);
  l_gauge_fix        (lambdaA,mpsB_imag);

  world.gop.fence();
//std::cout << "DEBUG[" << world.rank() << "] : 07" << std::endl;

  r_gauge_fix_inverse(lambdaB,mpsB_real);
  r_gauge_fix_inverse(lambdaB,mpsB_imag);

  world.gop.fence();
//std::cout << "DEBUG[" << world.rank() << "] : 08" << std::endl;

//std::cout << "DEBUG[" << world.rank() << "] : FF" << std::endl;
//return -log(sgvNorm2)/wfnNorm2/dt/2.0; // FIXME: this is wrong... what value should be returned?
  return sgvNorm2/wfnNorm2; // this must be 1.0?
}
