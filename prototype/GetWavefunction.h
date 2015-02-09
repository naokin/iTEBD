#ifndef __PROTOTYPE_GET_WAVEFUNCTION_H
#define __PROTOTYPE_GET_WAVEFUNCTION_H

#include "MPS.h"
#include "Wavefunction.h"

void GetWavefunction (const MPS<double>& A, const MPS<double>& B, Wavefunction<double>& Wfn);

#endif // __PROTOTYPE_GET_WAVEFUNCTION_H
