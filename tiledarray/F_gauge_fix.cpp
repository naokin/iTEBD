#include "F_gauge_fix.h"

/// Compute g * MPS
/// \param g left gauge matrix (only diagonal elements are stored)
/// \param mps MPS
void l_gauge_fix (const std::vector<double>& g, MPS<double>& mps)
{
  for(auto it = mps.matrix_u.begin(); it != mps.matrix_u.end(); ++it) {
    auto ranges = it.make_range();
    size_t lrow = ranges.start()[0];
    size_t urow = ranges.finish()[0];
    size_t dcol = ranges.size()[1];
    auto xt = it->get().begin();
    for(size_t ix = lrow; ix < urow; ++ix)
      for(size_t jx = 0; jx < dcol; ++jx, ++xt) *xt *= g[ix];
  }

  for(auto it = mps.matrix_d.begin(); it != mps.matrix_d.end(); ++it) {
    auto ranges = it.make_range();
    size_t lrow = ranges.start()[0];
    size_t urow = ranges.finish()[0];
    size_t dcol = ranges.size()[1];
    auto xt = it->get().begin();
    for(size_t ix = lrow; ix < urow; ++ix)
      for(size_t jx = 0; jx < dcol; ++jx, ++xt) *xt *= g[ix];
  }
}

void r_gauge_fix (const std::vector<double>& g, MPS<double>& mps)
{
  for(auto it = mps.matrix_u.begin(); it != mps.matrix_u.end(); ++it) {
    auto ranges = it.make_range();
    size_t drow = ranges.size()[0];
    size_t lcol = ranges.start()[1];
    size_t ucol = ranges.finish()[1];
    auto xt = it->get().begin();
    for(size_t ix = 0; ix < drow; ++ix)
      for(size_t jx = lcol; jx < ucol; ++jx, ++xt) *xt *= g[jx];
  }

  for(auto it = mps.matrix_d.begin(); it != mps.matrix_d.end(); ++it) {
    auto ranges = it.make_range();
    size_t drow = ranges.size()[0];
    size_t lcol = ranges.start()[1];
    size_t ucol = ranges.finish()[1];
    auto xt = it->get().begin();
    for(size_t ix = 0; ix < drow; ++ix)
      for(size_t jx = lcol; jx < ucol; ++jx, ++xt) *xt *= g[jx];
  }
}

/// Compute g^-1 * MPS
void l_gauge_fix_inverse (const std::vector<double>& g, MPS<double>& mps)
{
  for(auto it = mps.matrix_u.begin(); it != mps.matrix_u.end(); ++it) {
    auto ranges = it.make_range();
    size_t lrow = ranges.start()[0];
    size_t urow = ranges.finish()[0];
    size_t dcol = ranges.size()[1];
    auto xt = it->get().begin();
    for(size_t ix = lrow; ix < urow; ++ix)
      for(size_t jx = 0; jx < dcol; ++jx, ++xt) *xt /= g[ix];
  }

  for(auto it = mps.matrix_d.begin(); it != mps.matrix_d.end(); ++it) {
    auto ranges = it.make_range();
    size_t lrow = ranges.start()[0];
    size_t urow = ranges.finish()[0];
    size_t dcol = ranges.size()[1];
    auto xt = it->get().begin();
    for(size_t ix = lrow; ix < urow; ++ix)
      for(size_t jx = 0; jx < dcol; ++jx, ++xt) *xt /= g[ix];
  }
}

void r_gauge_fix_inverse (const std::vector<double>& g, MPS<double>& mps)
{
  for(auto it = mps.matrix_u.begin(); it != mps.matrix_u.end(); ++it) {
    auto ranges = it.make_range();
    size_t drow = ranges.size()[0];
    size_t lcol = ranges.start()[1];
    size_t ucol = ranges.finish()[1];
    auto xt = it->get().begin();
    for(size_t ix = 0; ix < drow; ++ix)
      for(size_t jx = lcol; jx < ucol; ++jx, ++xt) *xt /= g[jx];
  }

  for(auto it = mps.matrix_d.begin(); it != mps.matrix_d.end(); ++it) {
    auto ranges = it.make_range();
    size_t drow = ranges.size()[0];
    size_t lcol = ranges.start()[1];
    size_t ucol = ranges.finish()[1];
    auto xt = it->get().begin();
    for(size_t ix = 0; ix < drow; ++ix)
      for(size_t jx = lcol; jx < ucol; ++jx, ++xt) *xt /= g[jx];
  }
}
