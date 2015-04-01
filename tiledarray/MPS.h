#ifndef __TA_iTEBD_MPS_HPP
#define __TA_iTEBD_MPS_HPP

#include <tiledarray.h>

/// This is just a matrix container
/// Note that index of sparse block must be determined externally
template<typename T>
struct MPS {

  typedef TiledArray::Tensor<T> tile_t;

  typedef TiledArray::SparsePolicy policy;

  typedef TiledArray::Array<T,2,tile_t,policy> matrix_t;

  /// S = +1
  matrix_t matrix_u;

  /// S = -1
  matrix_t matrix_d;

};

#endif // __iTEBD_MPS_HPP
