#include "GetWavefunction.h"

void GetWavefunction (const MPS<double>& A, const MPS<double>& B, Wavefunction<double>& Wfn)
{
  Wfn.matrix_uu("i,j") = A.matrix_u("i,k")*B.matrix_u("k,j");
  Wfn.matrix_ud("i,j") = A.matrix_u("i,k")*B.matrix_d("k,j");
  Wfn.matrix_du("i,j") = A.matrix_d("i,k")*B.matrix_u("k,j");
  Wfn.matrix_dd("i,j") = A.matrix_d("i,k")*B.matrix_d("k,j");
}
