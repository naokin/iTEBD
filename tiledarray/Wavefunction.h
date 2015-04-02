#ifndef __TA_iTEBD_WAVEFUNCTION_HPP
#define __TA_iTEBD_WAVEFUNCTION_HPP

#include <tiledarray.h>

/// This is just a matrix container
/// Note that index of sparse block must be determined externally
template<typename T>
struct Wavefunction {

  typedef TiledArray::Tensor<T> tile_t;

  typedef TiledArray::SparsePolicy policy;

  typedef TiledArray::Array<T,2,tile_t,policy> matrix_t;

  /// S = +1,+1
  matrix_t matrix_uu;

  /// S = +1,-1
  matrix_t matrix_ud;

  /// S = -1,+1
  matrix_t matrix_du;

  /// S = -1,-1
  matrix_t matrix_dd;

};

#endif // __TA_iTEBD_WAVEFUNCTION_HPP
