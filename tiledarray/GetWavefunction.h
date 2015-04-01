#ifndef __TA_GET_WAVEFUNCTION_H
#define __TA_GET_WAVEFUNCTION_H

#include "MPS.h"
#include "Wavefunction.h"

void GetWavefunction (const MPS<double>& A, const MPS<double>& B, Wavefunction<double>& Wfn);

#endif // __TA_GET_WAVEFUNCTION_H
