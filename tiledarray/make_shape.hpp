#ifndef __TA_MAKE_SHAPE_HPP
#define __TA_MAKE_SHAPE_HPP

#include <tiledarray.h>
#include <vector>

template<class Q>
void TA_make_shape_helper (size_t& idx, TiledArray::Tensor<float>& shape, const Q& q0, const std::vector<Q>& q1)
{
  for(size_t i = 0; i < q1.size(); ++i, ++idx) if(q0 == q1[i]) shape[idx] = 1.0;
}

template<class Q, class... Args>
void TA_make_shape_helper (size_t& idx, TiledArray::Tensor<float>& shape, const Q& q0, const std::vector<Q>& q1, const Args&... qr)
{
  for(size_t i = 0; i < q1.size(); ++i) TA_make_shape_helper(idx,shape,q0+q1[i],qr...);
}

template<class Q, class... Args>
TiledArray::SparseShape<float> make_shape (const TiledArray::TiledRange& tr, const Q& q0, const std::vector<Q>& q1, const Args&... qr)
{
  namespace TA = TiledArray;

  TiledArray::Tensor<float> shape(tr.tiles(),0.0);

  size_t idx = 0; TA_make_shape_helper(idx,shape,q0,q1,qr...);

  return TiledArray::SparseShape<float>(shape,tr);
}

#endif // __TA_MAKE_SHAPE_HPP
