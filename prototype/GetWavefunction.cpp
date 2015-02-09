#include "GetWavefunction.h"
#include "MatrixFunctions.h"

void GetWavefunction (const MPS<double>& A, const MPS<double>& B, Wavefunction<double>& Wfn)
{
  MatrixMultiply(A.matrix_u,B.matrix_u,Wfn.matrix_uu);
  MatrixMultiply(A.matrix_u,B.matrix_d,Wfn.matrix_ud);
  MatrixMultiply(A.matrix_d,B.matrix_u,Wfn.matrix_du);
  MatrixMultiply(A.matrix_d,B.matrix_d,Wfn.matrix_dd);
}
