#ifndef __iTEBD_MPS_HPP
#define __iTEBD_MPS_HPP

#include "MatrixFunctions.h"

/// This is just a matrix container
/// Note that index of sparse block must be determined externally
template<typename T>
struct MPS {

  /// S = +1
  matrix_type<T> matrix_u;

  /// S = -1
  matrix_type<T> matrix_d;

};

#endif // __iTEBD_MPS_HPP
