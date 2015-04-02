#include "TA_sparse_svd.h"

#include <iostream>
#include <iomanip>

#include <cassert>
#include <cmath>
#include <cstring>

#include <Eigen/SVD>

#define _Q_NOT_FOUND_ 0x80000000

/// Supportive function to find index specified by quantum number
size_t F_find_index_quanta (const std::vector<int>& qX, int qValue)
{
  auto it = std::lower_bound(qX.begin(),qX.end(),qValue);
  if(it != qX.end() && (*it) == qValue)
    return std::distance(qX.begin(),it);
  else
    return _Q_NOT_FOUND_;
}

void TA_sparse_svd (
      madness::World& world,
const std::vector<int>& qR, /// Spin quantum #s for row index (replicated)
const std::vector<int>& qC, /// Spin quantum #s for col index (replicated)
const Wavefunction<double>& wfn,
      std::vector<int>& qS, /// Spin quantum #s for selected singular values
      std::vector<double>& lambda,
      MPS<double>& mpsA,
      MPS<double>& mpsB,
      double CUTOFF_)
{
  namespace TA = TiledArray;

  // this is also used for static check whether MPS<double>::matrix_t == Wavefunction<double>::matrix_t == matrix_type
  typedef TA::Array<double,2,TA::Tensor<double>,TA::SparsePolicy> matrix_type;

  typedef std::vector<size_t> index_type;

  // world info
  size_t nproc = world.size();
  size_t iproc = world.rank();

  // define reference for convenience
  // matrix_type<double> -> TA::Array<double>

  const matrix_type& uu = wfn.matrix_uu;
  const matrix_type& ud = wfn.matrix_ud;
  const matrix_type& du = wfn.matrix_du;
  const matrix_type& dd = wfn.matrix_dd;

  // calculate quantum#s of non-truncated singular value sectors

  std::vector<int> q0; q0.reserve(2*qR.size());

  for(size_t i = 0; i < qR.size(); ++i) {
    q0.push_back(qR[i]+1);
    q0.push_back(qR[i]-1);
  }

  // remove duplication...

  std::sort(q0.begin(),q0.end());
  q0.resize(std::distance(q0.begin(),std::unique(q0.begin(),q0.end())));

  size_t q0size = q0.size();

  // storage for singular values

  std::vector<std::vector<double>> tmp_lm(q0size);

  // storage for truncated singular vectors

  std::vector<std::unique_ptr<TA::Tensor<double>>> tmp_ua;
  std::vector<std::unique_ptr<TA::Tensor<double>>> tmp_da;
  std::vector<std::unique_ptr<TA::Tensor<double>>> tmp_ub;
  std::vector<std::unique_ptr<TA::Tensor<double>>> tmp_db;

  // stored locally

  tmp_ua.resize(q0size);
  tmp_da.resize(q0size);
  tmp_ub.resize(q0size);
  tmp_db.resize(q0size);

  // number of selected quanta
  size_t nSelQ = 0;
  // number of selected states
  size_t nSelM = 0;

  TA::TiledRange1 rangeR = uu.trange().data()[0];
  TA::TiledRange1 rangeC = uu.trange().data()[1];

  // Loop over symmetry of singular values

  std::vector<size_t> nSvals(q0size,0);
  std::vector<size_t> iOwner(q0size,0);

  for(size_t k = 0; k < q0size; ++k) {

    // Def. bitShape for merged matrix
    // (0)  (4)  (f)  (2)
    // 0 0  0 1  1 1  0 0
    // 0 0, 0 0, 1 1, 1 0
    unsigned char bitShape = 0x0;

    size_t nrow = 0;
    size_t ncol = 0;

    // offset
    size_t prow = 0;
    size_t pcol = 0;

    // find spin symmetry
    // e.g.)
    // qR = 0 2 4 6 8 10 ...
    // qC = 1 3 5 7 9 11 ...

    // index objects for sparse blocks
    size_t iu,ju,id,jd;

    iu = F_find_index_quanta(qR,q0[k]-1); // Find qR[:]+1 == q0[i]
    ju = F_find_index_quanta(qC,q0[k]+1); // Find -q0[i] == -qC[:]+1

    id = F_find_index_quanta(qR,q0[k]+1); // Find qR[:]-1 == q0[i]
    jd = F_find_index_quanta(qC,q0[k]-1); // Find -q0[i] == -qC[:]+1

    index_type ijuu = {iu,ju};
    index_type ijud = {iu,jd};
    index_type ijdu = {id,ju};
    index_type ijdd = {id,jd};

    // working representation of sparse block
    matrix_type::value_type uu_rep;
    matrix_type::value_type ud_rep;
    matrix_type::value_type du_rep;
    matrix_type::value_type dd_rep;

    if(iu != _Q_NOT_FOUND_ && ju != _Q_NOT_FOUND_ && !uu.is_zero(ijuu)) {
      uu_rep = uu.find(ijuu);
      prow = uu_rep.range().size()[0];
      pcol = uu_rep.range().size()[1];
    //if((bitShape ^ 0xf) & 0x4) -- always true at this line
        nrow += prow;
    //if((bitShape ^ 0xf) & 0x2) -- always true at this line
        ncol += pcol;
      bitShape |= 0x8;
    }

    if(iu != _Q_NOT_FOUND_ && jd != _Q_NOT_FOUND_ && !ud.is_zero(ijud)) {
      ud_rep = ud.find(ijud);
      prow = ud_rep.range().size()[0];
      if((bitShape ^ 0xf) & 0x8)
        nrow += prow;
    //if((bitShape ^ 0xf) & 0x1) -- always true at this line
        ncol += ud_rep.range().size()[1];
      bitShape |= 0x4;
    }

    if(id != _Q_NOT_FOUND_ && ju != _Q_NOT_FOUND_ && !du.is_zero(ijdu)) {
      du_rep = du.find(ijdu);
      pcol = du_rep.range().size()[1];
    //if((bitShape ^ 0xf) & 0x1) -- always true at this line
        nrow += du_rep.range().size()[0];
      if((bitShape ^ 0xf) & 0x8)
        ncol += pcol;
      bitShape |= 0x2;
    }

    if(id != _Q_NOT_FOUND_ && jd != _Q_NOT_FOUND_ && !dd.is_zero(ijdd)) {
      dd_rep = dd.find(ijdd);
      if((bitShape ^ 0xf) & 0x2)
        nrow += dd_rep.range().size()[0];
      if((bitShape ^ 0xf) & 0x4)
        ncol += dd_rep.range().size()[1];
      bitShape |= 0x1;
    }

    world.gop.fence(); // FIXME: does this need?

    if(!bitShape || (k%nproc != iproc)) continue; // TODO: needs better parallel mapping?

    TA::EigenMatrixXd C = TA::EigenMatrixXd::Zero(nrow,ncol);

    if(bitShape & 0x8) {
      TA::EigenMatrixXd bf_rep(     prow,     pcol);
      memcpy(bf_rep.data(),uu_rep.data(),uu_rep.size()*sizeof(double));
      C.block(   0,   0,     prow,     pcol) = bf_rep;
    }
    if(bitShape & 0x4) {
      TA::EigenMatrixXd bf_rep(     prow,ncol-pcol);
      memcpy(bf_rep.data(),ud_rep.data(),ud_rep.size()*sizeof(double));
      C.block(   0,pcol,     prow,ncol-pcol) = bf_rep;
    }
    if(bitShape & 0x2) {
      TA::EigenMatrixXd bf_rep(nrow-prow,     pcol);
      memcpy(bf_rep.data(),du_rep.data(),du_rep.size()*sizeof(double));
      C.block(prow,   0,nrow-prow,     pcol) = bf_rep;
    }
    if(bitShape & 0x1) {
      TA::EigenMatrixXd bf_rep(nrow-prow,ncol-pcol);
      memcpy(bf_rep.data(),dd_rep.data(),dd_rep.size()*sizeof(double));
      C.block(prow,pcol,nrow-prow,ncol-pcol) = bf_rep;
    }

    // now having a merged blcok matrix

    Eigen::JacobiSVD<TA::EigenMatrixXd> svds(C,Eigen::ComputeThinU|Eigen::ComputeThinV);

    TA::EigenMatrixXd U = svds.matrixU();
    TA::EigenMatrixXd Vt= svds.matrixV().transpose();

    size_t kSvals = 0;
    for(; kSvals < svds.singularValues().size(); ++kSvals)
      if(svds.singularValues()[kSvals] < CUTOFF_) break;

    // No singular value is selected...
    if(kSvals == 0) continue;

    nSvals[k] = kSvals;
    iOwner[k] = iproc;

    tmp_lm[k].resize(kSvals);
    for(size_t kSel = 0; kSel < kSvals; ++kSel) tmp_lm[k][kSel] = svds.singularValues()[kSel];

    if(iu != _Q_NOT_FOUND_ && prow != 0) {
      tmp_ua[k].reset(new TA::Tensor<double>(TA::Range(     prow,kSvals)));
      TA::eigen_submatrix_to_tensor( U.block(   0,   0,     prow,kSvals),*tmp_ua[k]);
    }
    if(id != _Q_NOT_FOUND_ && prow != nrow) {
      tmp_da[k].reset(new TA::Tensor<double>(TA::Range(nrow-prow,kSvals)));
      TA::eigen_submatrix_to_tensor( U.block(prow,   0,nrow-prow,kSvals),*tmp_da[k]);
    }
    if(ju != _Q_NOT_FOUND_ && pcol != 0) {
      tmp_ub[k].reset(new TA::Tensor<double>(TA::Range(kSvals,     pcol)));
      TA::eigen_submatrix_to_tensor(Vt.block(   0,   0,kSvals,     pcol),*tmp_ub[k]);
    }
    if(jd != _Q_NOT_FOUND_ && pcol != ncol) {
      tmp_db[k].reset(new TA::Tensor<double>(TA::Range(kSvals,ncol-pcol)));
      TA::eigen_submatrix_to_tensor(Vt.block(   0,pcol,kSvals,ncol-pcol),*tmp_db[k]);
    }

    ++nSelQ; nSelM += kSvals;
  }

  world.gop.fence(); // FIXME: does this need?

  world.gop.sum(nSvals.data(),q0size);
  world.gop.sum(iOwner.data(),q0size);

  world.gop.sum(nSelQ);
  world.gop.sum(nSelM);

  std::vector<size_t> reindex(q0size,q0size);

  qS.resize(nSelQ);
  lambda.resize(nSelM);

  size_t iQ = 0;
  size_t iM = 0;

  std::vector<size_t> newRanges(1+nSelQ,0);

  for(size_t k = 0; k < q0size; ++k) {

    if(nSvals[k] == 0) continue;

    qS[iQ] = q0[k];
    reindex[k] = iQ;

    if(iOwner[k] == iproc)
      std::copy(tmp_lm[k].data(),tmp_lm[k].data()+nSvals[k],lambda.data()+iM);

    world.gop.broadcast(lambda.data()+iM,nSvals[k],iOwner[k]);

    ++iQ; iM += nSvals[k];

    newRanges[iQ] = iM;
  }

  // new range object

  TA::TiledRange1 rangeS(newRanges.begin(),newRanges.end());

  // new matrix shapes

  size_t qSsize = nSelQ;
  size_t qRsize = qR.size();
  size_t qCsize = qC.size();

  TA::Tensor<float> uShapeA(TA::Range(qRsize,qSsize),0.0);
  TA::Tensor<float> dShapeA(TA::Range(qRsize,qSsize),0.0);

  for(size_t i = 0; i < qRsize; ++i)
    for(size_t j = 0; j < qSsize; ++j) {
      if((qR[i]+1) == qS[j])
        uShapeA[i*qSsize+j] = 1.0;
      if((qR[i]-1) == qS[j])
        dShapeA[i*qSsize+j] = 1.0;
    }

  TA::Tensor<float> uShapeB(TA::Range(qSsize,qCsize),0.0);
  TA::Tensor<float> dShapeB(TA::Range(qSsize,qCsize),0.0);

  for(size_t i = 0; i < qSsize; ++i)
    for(size_t j = 0; j < qCsize; ++j) {
      if((qS[i]+1) == qC[j])
        uShapeB[i*qCsize+j] = 1.0;
      if((qS[i]-1) == qC[j])
        dShapeB[i*qCsize+j] = 1.0;
    }

  // construct Array objects for return

  std::vector<TA::TiledRange1> rangesA = {rangeR,rangeS};
  TA::TiledRange trangeA(rangesA.begin(),rangesA.end());

  mpsA.matrix_u = matrix_type(world,trangeA,TA::SparseShape<float>(uShapeA,trangeA));
  mpsA.matrix_d = matrix_type(world,trangeA,TA::SparseShape<float>(dShapeA,trangeA));

  std::vector<TA::TiledRange1> rangesB = {rangeS,rangeC};
  TA::TiledRange trangeB(rangesB.begin(),rangesB.end());

  mpsB.matrix_u = matrix_type(world,trangeB,TA::SparseShape<float>(uShapeB,trangeB));
  mpsB.matrix_d = matrix_type(world,trangeB,TA::SparseShape<float>(dShapeB,trangeB));

//mpsA.matrix_u.set_all_local(0.0);
//mpsA.matrix_d.set_all_local(0.0);
//mpsB.matrix_u.set_all_local(0.0);
//mpsB.matrix_d.set_all_local(0.0);

  for(size_t k = 0; k < q0size; ++k) {

    if(nSvals[k] == 0) continue;

    // index objects for sparse blocks
    size_t iu,ju,id,jd;

    iu = F_find_index_quanta(qR,q0[k]-1); // Find qR[:]+1 == q0[i]
    id = F_find_index_quanta(qR,q0[k]+1); // Find qR[:]-1 == q0[i]

    ju = F_find_index_quanta(qC,q0[k]+1); // Find q0[i] == qC[:]+1
    jd = F_find_index_quanta(qC,q0[k]-1); // Find q0[i] == qC[:]-1

    size_t ks = reindex[k];

    index_type iuks = {iu,ks};
    index_type idks = {id,ks};
    index_type ksju = {ks,ju};
    index_type ksjd = {ks,jd};

    world.gop.fence(); // this is critical

    if(iOwner[k] == iproc) {
      if(iu != _Q_NOT_FOUND_) {
        if(mpsA.matrix_u.is_local(iuks))
          mpsA.matrix_u.set(iuks,tmp_ua[k]->data());
        else
          world.gop.send(mpsA.matrix_u.owner(iuks),4*k+0,*tmp_ua[k]);
      }
      if(id != _Q_NOT_FOUND_) {
        if(mpsA.matrix_d.is_local(idks))
          mpsA.matrix_d.set(idks,tmp_da[k]->data());
        else
          world.gop.send(mpsA.matrix_d.owner(idks),4*k+1,*tmp_da[k]);
      }
      if(ju != _Q_NOT_FOUND_) {
        if(mpsB.matrix_u.is_local(ksju))
          mpsB.matrix_u.set(ksju,tmp_ub[k]->data());
        else
          world.gop.send(mpsB.matrix_u.owner(ksju),4*k+2,*tmp_ub[k]);
      }
      if(jd != _Q_NOT_FOUND_) {
        if(mpsB.matrix_d.is_local(ksjd))
          mpsB.matrix_d.set(ksjd,tmp_db[k]->data());
        else
          world.gop.send(mpsB.matrix_d.owner(ksjd),4*k+3,*tmp_db[k]);
      }
    }
    else {
      if(iu != _Q_NOT_FOUND_ && mpsA.matrix_u.is_local(iuks))
        mpsA.matrix_u.set(iuks,world.gop.recv<TA::Tensor<double>>(iOwner[k],4*k+0).get().data());

      if(id != _Q_NOT_FOUND_ && mpsA.matrix_d.is_local(idks))
        mpsA.matrix_d.set(idks,world.gop.recv<TA::Tensor<double>>(iOwner[k],4*k+1).get().data());

      if(ju != _Q_NOT_FOUND_ && mpsB.matrix_u.is_local(ksju))
        mpsB.matrix_u.set(ksju,world.gop.recv<TA::Tensor<double>>(iOwner[k],4*k+2).get().data());

      if(jd != _Q_NOT_FOUND_ && mpsB.matrix_d.is_local(ksjd))
        mpsB.matrix_d.set(ksjd,world.gop.recv<TA::Tensor<double>>(iOwner[k],4*k+3).get().data());
    }

    world.gop.fence(); // FIXME: does this need?
  }

}
