#include "SqNorm.h"

#include <iostream>
#include <iomanip>

double SqNorm (madness::World& world, const Wavefunction<double>& wfn)
{
  double nrm2 = wfn.matrix_uu("i,j").squared_norm() +
      wfn.matrix_ud("i,j").squared_norm() +
      wfn.matrix_du("i,j").squared_norm() +
      wfn.matrix_dd("i,j").squared_norm();

  return nrm2;
}
