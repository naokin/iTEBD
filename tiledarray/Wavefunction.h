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

  T norm2 (madness::World& world) const {

    T sqnrm = static_cast<T>(0);

    for(auto it = matrix_uu.begin(); it != matrix_uu.end(); ++it) {
      auto x = it->get(); // TileReference
      for(auto xt = x.begin(); xt != x.end(); ++xt) sqnrm += (*xt)*(*xt);
    }
    world.gop.fence(); // FIXME: does this need?

    for(auto it = matrix_ud.begin(); it != matrix_ud.end(); ++it) {
      auto x = it->get(); // TileReference
      for(auto xt = x.begin(); xt != x.end(); ++xt) sqnrm += (*xt)*(*xt);
    }
    world.gop.fence(); // FIXME: does this need?

    for(auto it = matrix_du.begin(); it != matrix_du.end(); ++it) {
      auto x = it->get(); // TileReference
      for(auto xt = x.begin(); xt != x.end(); ++xt) sqnrm += (*xt)*(*xt);
    }
    world.gop.fence(); // FIXME: does this need?

    for(auto it = matrix_dd.begin(); it != matrix_dd.end(); ++it) {
      auto x = it->get(); // TileReference
      for(auto xt = x.begin(); xt != x.end(); ++xt) sqnrm += (*xt)*(*xt);
    }
    world.gop.fence(); // FIXME: does this need?

    world.gop.sum(&sqnrm,1);
    world.gop.broadcast(sqnrm);

    return sqnrm;
  }

};

#endif // __TA_iTEBD_WAVEFUNCTION_HPP
