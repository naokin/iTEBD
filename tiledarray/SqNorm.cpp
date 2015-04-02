#include "SqNorm.h"

#include <iostream>
#include <iomanip>

double SqNorm (madness::World& world, const Wavefunction<double>& wfn)
{
  double nrm2 = 0.0;

  for(size_t i = 0; i < wfn.matrix_uu.trange().tiles().volume(); ++i)
    if(!wfn.matrix_uu.is_zero(i) && wfn.matrix_uu.is_local(i))
      nrm2 += wfn.matrix_uu.find(i).get().squared_norm();

  for(size_t i = 0; i < wfn.matrix_ud.trange().tiles().volume(); ++i)
    if(!wfn.matrix_ud.is_zero(i) && wfn.matrix_ud.is_local(i))
      nrm2 += wfn.matrix_ud.find(i).get().squared_norm();

  for(size_t i = 0; i < wfn.matrix_du.trange().tiles().volume(); ++i)
    if(!wfn.matrix_du.is_zero(i) && wfn.matrix_du.is_local(i))
      nrm2 += wfn.matrix_du.find(i).get().squared_norm();

  for(size_t i = 0; i < wfn.matrix_dd.trange().tiles().volume(); ++i)
    if(!wfn.matrix_dd.is_zero(i) && wfn.matrix_dd.is_local(i))
      nrm2 += wfn.matrix_dd.find(i).get().squared_norm();

  world.gop.sum(&nrm2,1);

  return nrm2;
}
