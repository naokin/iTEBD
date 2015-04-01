#include "MPS_init.h"

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

  qA.clear(); qA.reserve(M_spin+1);
  qB.clear(); qB.reserve(M_spin+1);

  // define spin symmetry sector

  for(int k = -M_spin  ; k <= M_spin; k+=2) qA.push_back(k);
  for(int k = -M_spin+1; k <  M_spin; k+=2) qB.push_back(k);

  size_t qAsize = qA.size();
  size_t qBsize = qB.size();

  // init lambda

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

  // matrix shape

  TA::Tensor<float> uShapeA(TA::Range(qBsize,qAsize),0.0);
  TA::Tensor<float> dShapeA(TA::Range(qBsize,qAsize),0.0);

  for(size_t i = 0; i < qBsize; ++i)
    for(size_t j = 0; j < qAsize; ++j) {
      if((qB[i]+1) == qA[j])
        uShapeA[i*qAsize+j] = 1.0;
      if((qB[i]-1) == qA[j])
        dShapeA[i*qAsize+j] = 1.0;
    }

  TA::Tensor<float> uShapeB(TA::Range(qAsize,qBsize),0.0);
  TA::Tensor<float> dShapeB(TA::Range(qAsize,qBsize),0.0);

  for(size_t i = 0; i < qAsize; ++i)
    for(size_t j = 0; j < qBsize; ++j) {
      if((qA[i]+1) == qB[j])
        uShapeB[i*qBsize+j] = 1.0;
      if((qA[i]-1) == qB[j])
        dShapeB[i*qBsize+j] = 1.0;
    }

  // make matrix

  mpsA.matrix_u = TA::Array<double,2,TA::Tensor<double>,TA::SparsePolicy>(
    world,aMatrixTRange,TA::SparseShape<float>(uShapeA,aMatrixTRange));
  mpsA.matrix_u.set_all_local(1.0);

  mpsA.matrix_d = TA::Array<double,2,TA::Tensor<double>,TA::SparsePolicy>(
    world,aMatrixTRange,TA::SparseShape<float>(dShapeA,aMatrixTRange));
  mpsA.matrix_d.set_all_local(1.0);

  mpsB.matrix_u = TA::Array<double,2,TA::Tensor<double>,TA::SparsePolicy>(
    world,bMatrixTRange,TA::SparseShape<float>(uShapeB,bMatrixTRange));
  mpsB.matrix_u.set_all_local(1.0);

  mpsB.matrix_d = TA::Array<double,2,TA::Tensor<double>,TA::SparsePolicy>(
    world,bMatrixTRange,TA::SparseShape<float>(dShapeB,bMatrixTRange));
  mpsB.matrix_d.set_all_local(1.0);
}
