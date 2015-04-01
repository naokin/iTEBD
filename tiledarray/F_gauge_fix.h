#ifndef __TA_F_GAUGE_FIX_H
#define __TA_F_GAUGE_FIX_H

#include <vector>
#include "MPS.h"

/// Compute g * MPS
/// \param g left gauge matrix (only diagonal elements are stored)
/// \param mps MPS
void l_gauge_fix (const std::vector<double>& g, MPS<double>& mps);

void r_gauge_fix (const std::vector<double>& g, MPS<double>& mps);

/// Compute g^-1 * MPS
void l_gauge_fix_inverse (const std::vector<double>& g, MPS<double>& mps);

void r_gauge_fix_inverse (const std::vector<double>& g, MPS<double>& mps);

#endif // __TA_F_GAUGE_FIX_H
