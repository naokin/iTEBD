#ifndef __iTEBD_WAVEFUNCTION_HPP
#define __iTEBD_WAVEFUNCTION_HPP

#include "MatrixFunctions.h"

/// This is just a matrix container
/// Note that index of sparse block must be determined externally
template<typename T>
struct Wavefunction {

  /// S = +1,+1
  matrix_type<T> matrix_uu;

  /// S = +1,-1
  matrix_type<T> matrix_ud;

  /// S = -1,+1
  matrix_type<T> matrix_du;

  /// S = -1,-1
  matrix_type<T> matrix_dd;

};

#endif // __iTEBD_WAVEFUNCTION_HPP
