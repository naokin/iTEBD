#include "F_gauge_fix.h"

enum gauge_op_dir {
  gauge_op_L, gauge_op_R
};

enum gauge_op_type {
  gauge_op_mult, gauge_op_divide
};

template <typename Tile, gauge_op_dir GaugeOpDir>
Tile
gauge_op(const Tile& in_tile, const std::vector<double>& gauge_vec)
{
  Tile out_tile(in_tile.range());
  auto in_data = in_tile.data();
  auto out_data = out_tile.data();

  auto lRow = out_tile.range().start()[0];
  auto uRow = out_tile.range().finish()[0];
  auto lCol = out_tile.range().start()[1];
  auto uCol = out_tile.range().finish()[1];
  assert(uRow - lRow == out_tile.range().size()[0]);
  assert(uCol - lCol == out_tile.range().size()[1]);
  size_t rcOrd = 0;
  for(auto r = lRow; r < uRow; ++r)
    for(auto c = lCol; c < uCol; ++c, ++rcOrd) {
      auto gauge_factor = (GaugeOpDir == gauge_op_L) ? gauge_vec[r] : gauge_vec[c];
      out_data[rcOrd] = in_data[rcOrd] * gauge_factor;
    }

  return out_tile;
}

template <gauge_op_dir GaugeOpDir, gauge_op_type GaugeOpType>
void gauge_fix (const std::vector<double>& g, MPS<double>& mps) {
  typedef MPS<double>::matrix_t matrix_type;
  typedef MPS<double>::matrix_t::eval_type tile_type;
  typedef madness::Future<tile_type> future_type;

  auto& world = mps.matrix_u.get_world();

  std::array<std::reference_wrapper<matrix_type>,2> mats{{mps.matrix_u, mps.matrix_d}};

  std::vector<double> g_inverse;
  if (GaugeOpType == gauge_op_divide) {
    g_inverse.resize(g.size());
    std::transform(g.begin(), g.end(), g_inverse.begin(), [](double x) -> double { return 1.0 / x; });
  }
  const std::vector<double>& gauge_vector = GaugeOpType == gauge_op_mult ? g : g_inverse;

  for(auto t=0; t!=2; ++t) {
    matrix_type& mat = mats[t];
    matrix_type new_mat(world, mat.trange(), mat.get_shape(), mat.get_pmap());
    for(auto it = std::begin(mat); it != std::end(mat); ++it) {
      future_type tile = *it;
      new_mat.set(it.ordinal(),
                world.taskq.add(gauge_op<tile_type,GaugeOpDir>,
                                tile,
                                gauge_vector
                               )
      );
    }
    mat = new_mat;
  }
}

/// Compute g * MPS
/// \param g left gauge matrix (only diagonal elements are stored)
/// \param mps MPS
void l_gauge_fix (const std::vector<double>& g, MPS<double>& mps)
{
  gauge_fix<gauge_op_L, gauge_op_mult>(g, mps);
}

void r_gauge_fix (const std::vector<double>& g, MPS<double>& mps)
{
  gauge_fix<gauge_op_R, gauge_op_mult>(g, mps);
}

/// Compute g^-1 * MPS
void l_gauge_fix_inverse (const std::vector<double>& g, MPS<double>& mps)
{
  gauge_fix<gauge_op_L, gauge_op_divide>(g, mps);
}

void r_gauge_fix_inverse (const std::vector<double>& g, MPS<double>& mps)
{
  gauge_fix<gauge_op_R, gauge_op_divide>(g, mps);
}
