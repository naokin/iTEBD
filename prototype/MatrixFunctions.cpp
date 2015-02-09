#include "MatrixFunctions.h"

void DeepCopy (const matrix_type<double>& x, matrix_type<double>& y)
{
  size_t nrow = x.rows();
  size_t ncol = x.cols();

  y = matrix_type<double>::Constant(nrow,ncol,0);

  for(size_t i = 0; i < nrow; ++i)
    for(size_t j = 0; j < ncol; ++j)
      if(x(i,j)) y(i,j).reset(new local_matrix_type<double>(*x(i,j)));
}

double SquareNorm (const matrix_type<double>& x)
{
  double value = 0.0;

  for(size_t i = 0; i < x.rows(); ++i)
    for(size_t j = 0; j < x.cols(); ++j)
      if(x(i,j)) {
        const local_matrix_type<double>& xij = *x(i,j);
        for(size_t ix = 0; ix < xij.rows(); ++ix)
          for(size_t jx = 0; jx < xij.cols(); ++jx)
            value += xij(ix,jx)*xij(ix,jx);
      }

  return value;
}

double DotProduct (const matrix_type<double>& x, const matrix_type<double>& y)
{
  assert(x.rows() == y.rows());
  assert(x.cols() == y.cols());

  double value = 0.0;

  for(size_t i = 0; i < x.rows(); ++i)
    for(size_t j = 0; j < x.cols(); ++j)
      if(x(i,j) && y(i,j)) {
        const local_matrix_type<double>& xij = *x(i,j);
        const local_matrix_type<double>& yij = *y(i,j);
        assert(xij.rows() == yij.rows());
        assert(xij.cols() == yij.cols());
        for(size_t ix = 0; ix < xij.rows(); ++ix)
          for(size_t jx = 0; jx < xij.cols(); ++jx)
            value += xij(ix,jx)*yij(ix,jx);
      }

  return value;
}

void Scale (const double& alpha, matrix_type<double>& x)
{
  for(size_t i = 0; i < x.rows(); ++i)
    for(size_t j = 0; j < x.cols(); ++j)
      if(x(i,j)) (*x(i,j)) *= alpha;
}

void ScaledAdd (const double& alpha, matrix_type<double>& x, matrix_type<double>& y)
{
  size_t nrow = x.rows(); assert(nrow == y.rows());
  size_t ncol = x.cols(); assert(ncol == y.cols());

  for(size_t i = 0; i < nrow; ++i)
    for(size_t j = 0; j < ncol; ++j)
      if(x(i,j)) {
        if(y(i,j))
          (*y(i,j)) += alpha * (*x(i,j));
        else
          y(i,j).reset(new local_matrix_type<double>(alpha * (*x(i,j))));
      }
}

void MatrixMultiply (const matrix_type<double>& a, const matrix_type<double>& b, matrix_type<double>& c)
{
  size_t nrow = a.rows();
  size_t ncol = b.cols();
  size_t kdim = a.cols(); assert(kdim == b.rows());

  c = matrix_type<double>::Constant(nrow,ncol,0);

  for(size_t i = 0; i < nrow; ++i)
    for(size_t j = 0; j < ncol; ++j)
      for(size_t k = 0; k < kdim; ++k)
        if(a(i,k) && b(k,j)) {
          if(c(i,j))
            (*c(i,j)) += (*a(i,k)) * (*b(k,j));
          else
            c(i,j).reset(new local_matrix_type<double>((*a(i,k)) * (*b(k,j))));
        }
}
