#include "MPS_init.h"
#include "make_shape.hpp"

/// initializing MPS
/// \param qA quantum numbers on the left side of matrix A
/// \param dA size array of each quantum number for qA
/// \param lambdaA diagonal elements of gauge on the left side of matrix A
/// \param mpsA a bundle of A matrices
/// \param qB quantum numbers on the left side of matrix B
/// \param dB size array of each quantum number for qB
/// \param lambdaB diagonal elements of gauge on the left side of matrix B
/// \param mpsB a bundle of B matrices
/// \param M_spin max. spin quantum number (2S)
/// \param M_state number of renormalized states for each symmetry sector (set all symmetries has the same dim.)
void MPS_init (
      madness::World& world,
      std::vector<int>& qA, std::vector<double>& lambdaA, MPS<double>& mpsA,
      std::vector<int>& qB, std::vector<double>& lambdaB, MPS<double>& mpsB,
      int M_spin, size_t M_state)
{
  namespace TA = TiledArray;

  // quantum numbers for -A-[qA]-B-

  qA.clear(); qA.reserve(M_spin+1);

  // quantum numbers for -B-[qB]-A-

  qB.clear(); qB.reserve(M_spin+1);

  // construct spin quantum vectors

  for(int k = M_spin+1; k >= -M_spin+1; k-=2) qA.push_back(k);

  for(int k = M_spin  ; k >= -M_spin  ; k-=2) qB.push_back(k);

  size_t qAsize = qA.size();

  size_t qBsize = qB.size();

  // init lambda; set 1.0 for all quanta

  lambdaA.resize(M_state*qAsize,1.0);

  lambdaB.resize(M_state*qBsize,1.0);

  // matrix range

  std::vector<size_t> aBlock; aBlock.reserve(qAsize+1);
  for(size_t i = 0; i <= M_state*qAsize; i += M_state) aBlock.push_back(i);

  std::vector<size_t> bBlock; bBlock.reserve(qBsize+1);
  for(size_t i = 0; i <= M_state*qBsize; i += M_state) bBlock.push_back(i);

  std::vector<TA::TiledRange1> aMatrixRange(2);
  aMatrixRange[0] = TA::TiledRange1(bBlock.begin(), bBlock.end());
  aMatrixRange[1] = TA::TiledRange1(aBlock.begin(), aBlock.end());

  std::vector<TA::TiledRange1> bMatrixRange(2);
  bMatrixRange[0] = TA::TiledRange1(aBlock.begin(), aBlock.end());
  bMatrixRange[1] = TA::TiledRange1(bBlock.begin(), bBlock.end());

  TA::TiledRange aMatrixTRange(aMatrixRange.begin(), aMatrixRange.end());
  TA::TiledRange bMatrixTRange(bMatrixRange.begin(), bMatrixRange.end());

  // make matrix

  mpsA.matrix_u = TA::Array<double,2,TA::Tensor<double>,TA::SparsePolicy>(world,aMatrixTRange,make_shape(aMatrixTRange,+1,qB,qA));
  mpsA.matrix_u.set_all_local(1.0);

  mpsA.matrix_d = TA::Array<double,2,TA::Tensor<double>,TA::SparsePolicy>(world,aMatrixTRange,make_shape(aMatrixTRange,-1,qB,qA));
  mpsA.matrix_d.set_all_local(1.0);

  mpsB.matrix_u = TA::Array<double,2,TA::Tensor<double>,TA::SparsePolicy>(world,bMatrixTRange,make_shape(bMatrixTRange,+1,qA,qB));
  mpsB.matrix_u.set_all_local(1.0);

  mpsB.matrix_d = TA::Array<double,2,TA::Tensor<double>,TA::SparsePolicy>(world,bMatrixTRange,make_shape(bMatrixTRange,-1,qA,qB));
  mpsB.matrix_d.set_all_local(1.0);
}

/// initializing MPS
/// \param qA quantum numbers on the left side of matrix A
/// \param dA size array of each quantum number for qA
/// \param lambdaA diagonal elements of gauge on the left side of matrix A
/// \param mpsA_real a bundle of A matrices (real part)
/// \param mpsA_imag a bundle of A matrices (imag part)
/// \param qB quantum numbers on the left side of matrix B
/// \param dB size array of each quantum number for qB
/// \param lambdaB diagonal elements of gauge on the left side of matrix B
/// \param mpsB_real a bundle of B matrices (real part)
/// \param mpsB_imag a bundle of B matrices (imag part)
/// \param M_spin max. spin quantum number (2S)
/// \param M_state number of renormalized states for each symmetry sector (set all symmetries has the same dim.)
void MPS_init (
      madness::World& world,
      std::vector<int>& qA, std::vector<double>& lambdaA, MPS<double>& mpsA_real, MPS<double>& mpsA_imag,
      std::vector<int>& qB, std::vector<double>& lambdaB, MPS<double>& mpsB_real, MPS<double>& mpsB_imag,
      int M_spin, size_t M_state)
{
  namespace TA = TiledArray;

  // quantum numbers for -A-[qA]-B-

  qA.clear(); qA.reserve(M_spin+1);

  // quantum numbers for -B-[qB]-A-

  qB.clear(); qB.reserve(M_spin+1);

  // construct spin quantum vectors

  for(int k = M_spin+1; k >= -M_spin+1; k-=2) qA.push_back(k);

  for(int k = M_spin  ; k >= -M_spin  ; k-=2) qB.push_back(k);

  size_t qAsize = qA.size();

  size_t qBsize = qB.size();

  // init lambda; set 1.0 for all quanta

  lambdaA.resize(M_state*qAsize,1.0);

  lambdaB.resize(M_state*qBsize,1.0);

  // matrix range

  std::vector<size_t> aBlock; aBlock.reserve(qAsize+1);
  for(size_t i = 0; i <= M_state*qAsize; i += M_state) aBlock.push_back(i);

  std::vector<size_t> bBlock; bBlock.reserve(qBsize+1);
  for(size_t i = 0; i <= M_state*qBsize; i += M_state) bBlock.push_back(i);

  std::vector<TA::TiledRange1> aMatrixRange(2);
  aMatrixRange[0] = TA::TiledRange1(bBlock.begin(), bBlock.end());
  aMatrixRange[1] = TA::TiledRange1(aBlock.begin(), aBlock.end());

  std::vector<TA::TiledRange1> bMatrixRange(2);
  bMatrixRange[0] = TA::TiledRange1(aBlock.begin(), aBlock.end());
  bMatrixRange[1] = TA::TiledRange1(bBlock.begin(), bBlock.end());

  TA::TiledRange aMatrixTRange(aMatrixRange.begin(), aMatrixRange.end());
  TA::TiledRange bMatrixTRange(bMatrixRange.begin(), bMatrixRange.end());

  // make matrix

  mpsA_real.matrix_u = TA::Array<double,2,TA::Tensor<double>,TA::SparsePolicy>(world,aMatrixTRange,make_shape(aMatrixTRange,+1,qB,qA));
  mpsA_real.matrix_u.set_all_local(1.0);

  mpsA_real.matrix_d = TA::Array<double,2,TA::Tensor<double>,TA::SparsePolicy>(world,aMatrixTRange,make_shape(aMatrixTRange,-1,qB,qA));
  mpsA_real.matrix_d.set_all_local(1.0);

  mpsA_imag.matrix_u = TA::Array<double,2,TA::Tensor<double>,TA::SparsePolicy>(world,aMatrixTRange,make_shape(aMatrixTRange,+1,qB,qA));
  mpsA_imag.matrix_u.set_all_local(0.0);

  mpsA_imag.matrix_d = TA::Array<double,2,TA::Tensor<double>,TA::SparsePolicy>(world,aMatrixTRange,make_shape(aMatrixTRange,-1,qB,qA));
  mpsA_imag.matrix_d.set_all_local(0.0);

  mpsB_real.matrix_u = TA::Array<double,2,TA::Tensor<double>,TA::SparsePolicy>(world,bMatrixTRange,make_shape(bMatrixTRange,+1,qA,qB));
  mpsB_real.matrix_u.set_all_local(1.0);

  mpsB_real.matrix_d = TA::Array<double,2,TA::Tensor<double>,TA::SparsePolicy>(world,bMatrixTRange,make_shape(bMatrixTRange,-1,qA,qB));
  mpsB_real.matrix_d.set_all_local(1.0);

  mpsB_imag.matrix_u = TA::Array<double,2,TA::Tensor<double>,TA::SparsePolicy>(world,bMatrixTRange,make_shape(bMatrixTRange,+1,qA,qB));
  mpsB_imag.matrix_u.set_all_local(0.0);

  mpsB_imag.matrix_d = TA::Array<double,2,TA::Tensor<double>,TA::SparsePolicy>(world,bMatrixTRange,make_shape(bMatrixTRange,-1,qA,qB));
  mpsB_imag.matrix_d.set_all_local(0.0);
}
