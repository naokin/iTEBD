#ifndef __MATRIX_FUNCTIONS_H
#define __MATRIX_FUNCTIONS_H

#include <memory>
#include <Eigen/Core>

template<typename T>
using local_matrix_type = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>;

template<typename T>
using matrix_type = Eigen::Matrix<std::shared_ptr<local_matrix_type<T>>,Eigen::Dynamic,Eigen::Dynamic>;

void DeepCopy (const matrix_type<double>& x, matrix_type<double>& y);

double SquareNorm (const matrix_type<double>& x);

double DotProduct (const matrix_type<double>& x, const matrix_type<double>& y);

void Scale (const double& alpha, matrix_type<double>& x);

void ScaledAdd (const double& alpha, matrix_type<double>& x, matrix_type<double>& y);

void MatrixMultiply (const matrix_type<double>& a, const matrix_type<double>& b, matrix_type<double>& c);

#endif // __MATRIX_FUNCTIONS_H
