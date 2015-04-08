#include "TA_sparse_svd.h"

#include <iostream>
#include <iomanip>

#include <cassert>
#include <complex>
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

//  world.gop.fence(); // FIXME: does this need?

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

/// perform SVD on complex wavefunction
/// real and imaginary parts are stored separately
void TA_complex_sparse_svd (
      madness::World& world,
const std::vector<int>& qR, /// Spin quantum #s for row index (replicated)
const std::vector<int>& qC, /// Spin quantum #s for col index (replicated)
const Wavefunction<double>& wfn_real,
const Wavefunction<double>& wfn_imag,
      std::vector<int>& qS, /// Spin quantum #s for selected singular values
      std::vector<double>& lambda,
      MPS<double>& mpsA_real,
      MPS<double>& mpsA_imag,
      MPS<double>& mpsB_real,
      MPS<double>& mpsB_imag,
      double CUTOFF_)
{
//std::cout << "DEBUG[" << world.rank() << "] : 00" << std::endl;
  namespace TA = TiledArray;

  // this is also used for static check whether MPS<double>::matrix_t == Wavefunction<double>::matrix_t == matrix_type
  typedef TA::Array<double,2,TA::Tensor<double>,TA::SparsePolicy> matrix_type;

  typedef Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic> Dense_matrix_type;

  typedef std::vector<size_t> index_type;

  // world info
  size_t nproc = world.size();
  size_t iproc = world.rank();

  // define reference for convenience

  const matrix_type& uur = wfn_real.matrix_uu;
  const matrix_type& udr = wfn_real.matrix_ud;
  const matrix_type& dur = wfn_real.matrix_du;
  const matrix_type& ddr = wfn_real.matrix_dd;

  const matrix_type& uui = wfn_imag.matrix_uu;
  const matrix_type& udi = wfn_imag.matrix_ud;
  const matrix_type& dui = wfn_imag.matrix_du;
  const matrix_type& ddi = wfn_imag.matrix_dd;

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

  std::vector<std::unique_ptr<TA::Tensor<double>>> tmp_uar;
  std::vector<std::unique_ptr<TA::Tensor<double>>> tmp_dar;
  std::vector<std::unique_ptr<TA::Tensor<double>>> tmp_ubr;
  std::vector<std::unique_ptr<TA::Tensor<double>>> tmp_dbr;

  std::vector<std::unique_ptr<TA::Tensor<double>>> tmp_uai;
  std::vector<std::unique_ptr<TA::Tensor<double>>> tmp_dai;
  std::vector<std::unique_ptr<TA::Tensor<double>>> tmp_ubi;
  std::vector<std::unique_ptr<TA::Tensor<double>>> tmp_dbi;

  // stored locally

  tmp_uar.resize(q0size);
  tmp_dar.resize(q0size);
  tmp_ubr.resize(q0size);
  tmp_dbr.resize(q0size);

  tmp_uai.resize(q0size);
  tmp_dai.resize(q0size);
  tmp_ubi.resize(q0size);
  tmp_dbi.resize(q0size);

  // number of selected quanta
  size_t nSelQ = 0;
  // number of selected states
  size_t nSelM = 0;

  // NOTE: rows and cols are the same for each matrix
  // row[uur] == row[udr] == row[dur] == row[ddr] == row[uui] == row[udi] == row[dui] == row[ddi]
  // col[uur] == col[udr] == col[dur] == col[ddr] == col[uui] == col[udi] == col[dui] == col[ddi]

  TA::TiledRange1 rangeR = uur.trange().data()[0];
  TA::TiledRange1 rangeC = uur.trange().data()[1];

  // Loop over symmetry of singular values

  std::vector<size_t> nSvals(q0size,0);
  std::vector<size_t> iOwner(q0size,0);

  for(size_t k = 0; k < q0size; ++k) {

//std::cout << "DEBUG[" << world.rank() << "] : 01-A" << std::endl;
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

    matrix_type::value_type uur_rep;
    matrix_type::value_type udr_rep;
    matrix_type::value_type dur_rep;
    matrix_type::value_type ddr_rep;

    matrix_type::value_type uui_rep;
    matrix_type::value_type udi_rep;
    matrix_type::value_type dui_rep;
    matrix_type::value_type ddi_rep;

    // NOTE: matrix sizes of real- and imag-parts must be the same

    if(iu != _Q_NOT_FOUND_ && ju != _Q_NOT_FOUND_ && !uur.is_zero(ijuu)) {
      uur_rep = uur.find(ijuu);
      uui_rep = uui.find(ijuu);
      prow = uur_rep.range().size()[0];
      pcol = uur_rep.range().size()[1];
    //if((bitShape ^ 0xf) & 0x4) -- always true at this line
        nrow += prow;
    //if((bitShape ^ 0xf) & 0x2) -- always true at this line
        ncol += pcol;
      bitShape |= 0x8;
    }

    if(iu != _Q_NOT_FOUND_ && jd != _Q_NOT_FOUND_ && !udr.is_zero(ijud)) {
      udr_rep = udr.find(ijud);
      udi_rep = udi.find(ijud);
      prow = udr_rep.range().size()[0];
      if((bitShape ^ 0xf) & 0x8)
        nrow += prow;
    //if((bitShape ^ 0xf) & 0x1) -- always true at this line
        ncol += udr_rep.range().size()[1];
      bitShape |= 0x4;
    }

    if(id != _Q_NOT_FOUND_ && ju != _Q_NOT_FOUND_ && !dur.is_zero(ijdu)) {
      dur_rep = dur.find(ijdu);
      dui_rep = dui.find(ijdu);
      pcol = dur_rep.range().size()[1];
    //if((bitShape ^ 0xf) & 0x1) -- always true at this line
        nrow += dur_rep.range().size()[0];
      if((bitShape ^ 0xf) & 0x8)
        ncol += pcol;
      bitShape |= 0x2;
    }

    if(id != _Q_NOT_FOUND_ && jd != _Q_NOT_FOUND_ && !ddr.is_zero(ijdd)) {
      ddr_rep = ddr.find(ijdd);
      ddi_rep = ddi.find(ijdd);
      if((bitShape ^ 0xf) & 0x2)
        nrow += ddr_rep.range().size()[0];
      if((bitShape ^ 0xf) & 0x4)
        ncol += ddr_rep.range().size()[1];
      bitShape |= 0x1;
    }

//  world.gop.fence(); // FIXME: does this need?
//std::cout << "DEBUG[" << world.rank() << "] : 01-B" << std::endl;

    // TODO: needs better parallel mapping and/or load-balancing
    if(!bitShape || (k%nproc != iproc)) continue;

    Dense_matrix_type C = Dense_matrix_type::Constant(nrow,ncol,std::complex<double>(0.0,0.0));

    if(bitShape & 0x8) {
      TA::EigenMatrixXd bf_rep(     prow,     pcol);
      memcpy(bf_rep.data(),uur_rep.data(),uur_rep.size()*sizeof(double));
      C.real().block(   0,   0,     prow,     pcol) = bf_rep;
      memcpy(bf_rep.data(),uui_rep.data(),uui_rep.size()*sizeof(double));
      C.imag().block(   0,   0,     prow,     pcol) = bf_rep;
    }
    if(bitShape & 0x4) {
      TA::EigenMatrixXd bf_rep(     prow,ncol-pcol);
      memcpy(bf_rep.data(),udr_rep.data(),udr_rep.size()*sizeof(double));
      C.real().block(   0,pcol,     prow,ncol-pcol) = bf_rep;
      memcpy(bf_rep.data(),udi_rep.data(),udi_rep.size()*sizeof(double));
      C.imag().block(   0,pcol,     prow,ncol-pcol) = bf_rep;
    }
    if(bitShape & 0x2) {
      TA::EigenMatrixXd bf_rep(nrow-prow,     pcol);
      memcpy(bf_rep.data(),dur_rep.data(),dur_rep.size()*sizeof(double));
      C.real().block(prow,   0,nrow-prow,     pcol) = bf_rep;
      memcpy(bf_rep.data(),dui_rep.data(),dui_rep.size()*sizeof(double));
      C.imag().block(prow,   0,nrow-prow,     pcol) = bf_rep;
    }
    if(bitShape & 0x1) {
      TA::EigenMatrixXd bf_rep(nrow-prow,ncol-pcol);
      memcpy(bf_rep.data(),ddr_rep.data(),ddr_rep.size()*sizeof(double));
      C.real().block(prow,pcol,nrow-prow,ncol-pcol) = bf_rep;
      memcpy(bf_rep.data(),ddi_rep.data(),ddi_rep.size()*sizeof(double));
      C.imag().block(prow,pcol,nrow-prow,ncol-pcol) = bf_rep;
    }

    // now having a merged blcok matrix

//std::cout << "DEBUG[" << world.rank() << "] : 01-C" << std::endl;
    // SVD seems to be bottle neck...
    Eigen::JacobiSVD<Dense_matrix_type> svds(C,Eigen::ComputeThinU|Eigen::ComputeThinV);
//std::cout << "DEBUG[" << world.rank() << "] : 01-D" << std::endl;

    Dense_matrix_type U = svds.matrixU();
    Dense_matrix_type Vt= svds.matrixV().transpose();

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
      tmp_uar[k].reset(new TA::Tensor<double>(TA::Range(     prow,kSvals)));
      TA::eigen_submatrix_to_tensor( U.real().block(   0,   0,     prow,kSvals),*tmp_uar[k]);
      tmp_uai[k].reset(new TA::Tensor<double>(TA::Range(     prow,kSvals)));
      TA::eigen_submatrix_to_tensor( U.imag().block(   0,   0,     prow,kSvals),*tmp_uai[k]);
    }
    if(id != _Q_NOT_FOUND_ && prow != nrow) {
      tmp_dar[k].reset(new TA::Tensor<double>(TA::Range(nrow-prow,kSvals)));
      TA::eigen_submatrix_to_tensor( U.real().block(prow,   0,nrow-prow,kSvals),*tmp_dar[k]);
      tmp_dai[k].reset(new TA::Tensor<double>(TA::Range(nrow-prow,kSvals)));
      TA::eigen_submatrix_to_tensor( U.imag().block(prow,   0,nrow-prow,kSvals),*tmp_dai[k]);
    }
    if(ju != _Q_NOT_FOUND_ && pcol != 0) {
      tmp_ubr[k].reset(new TA::Tensor<double>(TA::Range(kSvals,     pcol)));
      TA::eigen_submatrix_to_tensor(Vt.real().block(   0,   0,kSvals,     pcol),*tmp_ubr[k]);
      tmp_ubi[k].reset(new TA::Tensor<double>(TA::Range(kSvals,     pcol)));
      TA::eigen_submatrix_to_tensor(Vt.imag().block(   0,   0,kSvals,     pcol),*tmp_ubi[k]);
    }
    if(jd != _Q_NOT_FOUND_ && pcol != ncol) {
      tmp_dbr[k].reset(new TA::Tensor<double>(TA::Range(kSvals,ncol-pcol)));
      TA::eigen_submatrix_to_tensor(Vt.real().block(   0,pcol,kSvals,ncol-pcol),*tmp_dbr[k]);
      tmp_dbi[k].reset(new TA::Tensor<double>(TA::Range(kSvals,ncol-pcol)));
      TA::eigen_submatrix_to_tensor(Vt.imag().block(   0,pcol,kSvals,ncol-pcol),*tmp_dbi[k]);
    }

    ++nSelQ; nSelM += kSvals;
//std::cout << "DEBUG[" << world.rank() << "] : 01-F" << std::endl;
  }

  world.gop.fence(); // FIXME: does this need?
//std::cout << "DEBUG[" << world.rank() << "] : 02" << std::endl;

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

  mpsA_real.matrix_u = matrix_type(world,trangeA,TA::SparseShape<float>(uShapeA,trangeA));
  mpsA_real.matrix_d = matrix_type(world,trangeA,TA::SparseShape<float>(dShapeA,trangeA));

  mpsA_imag.matrix_u = matrix_type(world,trangeA,TA::SparseShape<float>(uShapeA,trangeA));
  mpsA_imag.matrix_d = matrix_type(world,trangeA,TA::SparseShape<float>(dShapeA,trangeA));

  std::vector<TA::TiledRange1> rangesB = {rangeS,rangeC};
  TA::TiledRange trangeB(rangesB.begin(),rangesB.end());

  mpsB_real.matrix_u = matrix_type(world,trangeB,TA::SparseShape<float>(uShapeB,trangeB));
  mpsB_real.matrix_d = matrix_type(world,trangeB,TA::SparseShape<float>(dShapeB,trangeB));

  mpsB_imag.matrix_u = matrix_type(world,trangeB,TA::SparseShape<float>(uShapeB,trangeB));
  mpsB_imag.matrix_d = matrix_type(world,trangeB,TA::SparseShape<float>(dShapeB,trangeB));

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
//std::cout << "DEBUG[" << world.rank() << "] : 03" << std::endl;

    // Here, making a lot of communications

    if(iOwner[k] == iproc) {
      if(iu != _Q_NOT_FOUND_) {
        if(mpsA_real.matrix_u.is_local(iuks))
          mpsA_real.matrix_u.set(iuks,tmp_uar[k]->data());
        else
          world.gop.send(mpsA_real.matrix_u.owner(iuks),8*k+0,*tmp_uar[k]);

        if(mpsA_imag.matrix_u.is_local(iuks))
          mpsA_imag.matrix_u.set(iuks,tmp_uai[k]->data());
        else
          world.gop.send(mpsA_imag.matrix_u.owner(iuks),8*k+4,*tmp_uai[k]);
      }
//std::cout << "DEBUG[" << world.rank() << "] : 03-A" << std::endl;
      if(id != _Q_NOT_FOUND_) {
        if(mpsA_real.matrix_d.is_local(idks))
          mpsA_real.matrix_d.set(idks,tmp_dar[k]->data());
        else
          world.gop.send(mpsA_real.matrix_d.owner(idks),8*k+1,*tmp_dar[k]);

        if(mpsA_imag.matrix_d.is_local(idks))
          mpsA_imag.matrix_d.set(idks,tmp_dai[k]->data());
        else
          world.gop.send(mpsA_imag.matrix_d.owner(idks),8*k+5,*tmp_dai[k]);
      }
//std::cout << "DEBUG[" << world.rank() << "] : 03-B" << std::endl;
      if(ju != _Q_NOT_FOUND_) {
        if(mpsB_real.matrix_u.is_local(ksju))
          mpsB_real.matrix_u.set(ksju,tmp_ubr[k]->data());
        else
          world.gop.send(mpsB_real.matrix_u.owner(ksju),8*k+2,*tmp_ubr[k]);

        if(mpsB_imag.matrix_u.is_local(ksju))
          mpsB_imag.matrix_u.set(ksju,tmp_ubi[k]->data());
        else
          world.gop.send(mpsB_imag.matrix_u.owner(ksju),8*k+6,*tmp_ubi[k]);
      }
//std::cout << "DEBUG[" << world.rank() << "] : 03-C" << std::endl;
      if(jd != _Q_NOT_FOUND_) {
        if(mpsB_real.matrix_d.is_local(ksjd))
          mpsB_real.matrix_d.set(ksjd,tmp_dbr[k]->data());
        else
          world.gop.send(mpsB_real.matrix_d.owner(ksjd),8*k+3,*tmp_dbr[k]);

        if(mpsB_imag.matrix_d.is_local(ksjd))
          mpsB_imag.matrix_d.set(ksjd,tmp_dbi[k]->data());
        else
          world.gop.send(mpsB_imag.matrix_d.owner(ksjd),8*k+7,*tmp_dbi[k]);
      }
//std::cout << "DEBUG[" << world.rank() << "] : 03-D" << std::endl;
    }
    else {
      if(iu != _Q_NOT_FOUND_ && mpsA_real.matrix_u.is_local(iuks))
        mpsA_real.matrix_u.set(iuks,world.gop.recv<TA::Tensor<double>>(iOwner[k],8*k+0).get().data());
//std::cout << "DEBUG[" << world.rank() << "] : 03-A recv1" << std::endl;

      if(iu != _Q_NOT_FOUND_ && mpsA_imag.matrix_u.is_local(iuks))
        mpsA_imag.matrix_u.set(iuks,world.gop.recv<TA::Tensor<double>>(iOwner[k],8*k+4).get().data());
//std::cout << "DEBUG[" << world.rank() << "] : 03-A recv2" << std::endl;

      if(id != _Q_NOT_FOUND_ && mpsA_real.matrix_d.is_local(idks))
        mpsA_real.matrix_d.set(idks,world.gop.recv<TA::Tensor<double>>(iOwner[k],8*k+1).get().data());
//std::cout << "DEBUG[" << world.rank() << "] : 03-B recv1" << std::endl;

      if(id != _Q_NOT_FOUND_ && mpsA_imag.matrix_d.is_local(idks))
        mpsA_imag.matrix_d.set(idks,world.gop.recv<TA::Tensor<double>>(iOwner[k],8*k+5).get().data());
//std::cout << "DEBUG[" << world.rank() << "] : 03-B recv2" << std::endl;

      if(ju != _Q_NOT_FOUND_ && mpsB_real.matrix_u.is_local(ksju))
        mpsB_real.matrix_u.set(ksju,world.gop.recv<TA::Tensor<double>>(iOwner[k],8*k+2).get().data());
//std::cout << "DEBUG[" << world.rank() << "] : 03-C recv1" << std::endl;

      if(ju != _Q_NOT_FOUND_ && mpsB_imag.matrix_u.is_local(ksju))
        mpsB_imag.matrix_u.set(ksju,world.gop.recv<TA::Tensor<double>>(iOwner[k],8*k+6).get().data());
//std::cout << "DEBUG[" << world.rank() << "] : 03-C recv2" << std::endl;

      if(jd != _Q_NOT_FOUND_ && mpsB_real.matrix_d.is_local(ksjd))
        mpsB_real.matrix_d.set(ksjd,world.gop.recv<TA::Tensor<double>>(iOwner[k],8*k+3).get().data());
//std::cout << "DEBUG[" << world.rank() << "] : 03-D recv1" << std::endl;

      if(jd != _Q_NOT_FOUND_ && mpsB_imag.matrix_d.is_local(ksjd))
        mpsB_imag.matrix_d.set(ksjd,world.gop.recv<TA::Tensor<double>>(iOwner[k],8*k+7).get().data());
//std::cout << "DEBUG[" << world.rank() << "] : 03-D recv2" << std::endl;
    }

//std::cout << "DEBUG[" << world.rank() << "] : 03-F" << std::endl;
    world.gop.fence(); // FIXME: does this need?
//std::cout << "DEBUG[" << world.rank() << "] : 04" << std::endl;
  }

//std::cout << "DEBUG[" << world.rank() << "] : FF" << std::endl;
}
