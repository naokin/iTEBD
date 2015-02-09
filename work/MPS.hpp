#ifndef __iTEBD_MPS_HPP
#define __iTEBD_MPS_HPP

#include <tiledarray.h>

template<typename T>
struct MPS {

  /// type of block-sparse matrix
  typedef TiledArray::Array<T,2> matrix_type;

  /// type of each block matrix (i.e. type of tile)
  typedef typename matrix_type::value_type local_matrix_type;

  /// Matrix w/ Sz = +1
  matrix_type u_;

  /// Matrix w/ Sz = -1
  matrix_type d_;

};

#endif // __iTEBD_MPS_HPP
