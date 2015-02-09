#include "SparseSVD.h"
#include <iostream>
#include <iomanip>

#include <cassert>
#include <cmath>

#include <Eigen/SVD>

int findQuantaIndex (const std::vector<int>& qX, int qValue)
{
  auto it = std::lower_bound(qX.begin(),qX.end(),qValue);
  if(it != qX.end() && (*it) == qValue)
    return std::distance(qX.begin(),it);
  else
    return -1;
}

void SparseSVD (
  const std::vector<int>& qR, // Spin quantum #s for row index
  const std::vector<int>& qC, // Spin quantum #s for col index
  const Wavefunction<double>& Wfn,
        std::vector<int>& qS, // Spin quantum #s for selected singular values
        std::vector<std::vector<double>>& lambda,
        MPS<double>& aMps,
        MPS<double>& bMps,
        double CUTOFF_)
{
  // define reference for convenience

  const matrix_type<double>& uu = Wfn.matrix_uu;
  const matrix_type<double>& ud = Wfn.matrix_ud;
  const matrix_type<double>& du = Wfn.matrix_du;
  const matrix_type<double>& dd = Wfn.matrix_dd;

  // Calculating qS0 from row quanta
  std::vector<int> qS0; qS0.reserve(2*qR.size());

  for(size_t i = 0; i < qR.size(); ++i) {
    qS0.push_back(qR[i]+1);
    qS0.push_back(qR[i]-1);
  }
  // Remove duplication...
  std::sort(qS0.begin(),qS0.end());
  qS0.resize(std::distance(qS0.begin(),std::unique(qS0.begin(),qS0.end())));

  qS.clear();
  qS.reserve(qS0.size());

  lambda.clear();
  lambda.reserve(qS0.size());

  MPS<double> aMps_;
  aMps_.matrix_u.resize(qR.size(),qS0.size());
  aMps_.matrix_d.resize(qR.size(),qS0.size());

  MPS<double> bMps_;
  bMps_.matrix_u.resize(qS0.size(),qC.size());
  bMps_.matrix_d.resize(qS0.size(),qC.size());

  matrix_type<double>& ua = aMps_.matrix_u;
  matrix_type<double>& da = aMps_.matrix_d;
  matrix_type<double>& ub = bMps_.matrix_u;
  matrix_type<double>& db = bMps_.matrix_d;

  size_t nSym = 0; // Total # of qunatum blocks to be seletected
  size_t nSel = 0; // Total # of states

  // Loop over symmetry of singular values

  for(size_t k = 0; k < qS0.size(); ++k) {

//  std::cout << "\t\t\tSymmetry sector " << std::setw(4) << k << " (" << std::setw(4) << qS0[k] << ")" << std::endl;

    // (0)  (4)  (f)  (2)
    // 0 0  0 1  1 1  0 0
    // 0 0, 0 0, 1 1, 1 0
    unsigned char bitShape = 0x0;

    int nrow = 0;
    int ncol = 0;

    // offset
    int prow = 0;
    int pcol = 0;

    // find spin symmetry viewed as forward...

    // e.g.)
    // cQ = 0 2 4 6 8 10 ...
    // rQ = 1 3 5 7 9 11 ...

    int iu,ju,id,jd;

    iu = findQuantaIndex(qR,qS0[k]-1); // Find qR[:]+1 == qS0[i]
    ju = findQuantaIndex(qC,qS0[k]+1); // Find -qS0[i] == -qC[:]+1

    id = findQuantaIndex(qR,qS0[k]+1); // Find qR[:]-1 == qS0[i]
    jd = findQuantaIndex(qC,qS0[k]-1); // Find -qS0[i] == -qC[:]+1

    if(iu >= 0 && ju >= 0 && uu(iu,ju)) {
      prow = uu(iu,ju)->rows();
      pcol = uu(iu,ju)->cols();
    //if((bitShape ^ 0xf) & 0x4) -- always true at this line
        nrow += prow;
    //if((bitShape ^ 0xf) & 0x2) -- always true at this line
        ncol += pcol;
      bitShape |= 0x8;
    }

    if(iu >= 0 && jd >= 0 && ud(iu,jd)) {
      prow = ud(iu,jd)->rows();
      if((bitShape ^ 0xf) & 0x8)
        nrow += prow;
    //if((bitShape ^ 0xf) & 0x1) -- always true at this line
        ncol += ud(iu,jd)->cols();
      bitShape |= 0x4;
    }

    if(id >= 0 && ju >= 0 && du(id,ju)) {
      pcol = du(id,ju)->cols();
    //if((bitShape ^ 0xf) & 0x1) -- always true at this line
        nrow += du(id,ju)->rows();
      if((bitShape ^ 0xf) & 0x8)
        ncol += pcol;
      bitShape |= 0x2;
    }

    if(id >= 0 && jd >= 0 && dd(id,jd)) {
      if((bitShape ^ 0xf) & 0x2)
        nrow += dd(id,jd)->rows();
      if((bitShape ^ 0xf) & 0x4)
        ncol += dd(id,jd)->cols();
      bitShape |= 0x1;
    }

    if(!bitShape) continue;

    local_matrix_type<double> C = local_matrix_type<double>::Zero(nrow,ncol);

    if(bitShape & 0x8)
      C.block(   0,   0,     prow,     pcol) = *uu(iu,ju);
    if(bitShape & 0x4)
      C.block(   0,pcol,     prow,ncol-pcol) = *ud(iu,jd);
    if(bitShape & 0x2)
      C.block(prow,   0,nrow-prow,     pcol) = *du(id,ju);
    if(bitShape & 0x1)
      C.block(prow,pcol,nrow-prow,ncol-pcol) = *dd(id,jd);

    // now having a merged blcok matrix

//  double cNorm2 = 0.0;
//  for(int ix = 0; ix < C.rows(); ++ix)
//    for(int jx = 0; jx < C.cols(); ++jx)
//      cNorm2 += C(ix,jx)*C(ix,jx);

    // Ignore the case |C| is too small, to avoid numerical instability?
//  if(cNorm2 < 1.0e-16) continue;

    Eigen::JacobiSVD<local_matrix_type<double>> svds(C,Eigen::ComputeThinU|Eigen::ComputeThinV);

    local_matrix_type<double> U = svds.matrixU();
    local_matrix_type<double> Vt= svds.matrixV().transpose();

    int nSvd = 0;
    for(; nSvd < svds.singularValues().size(); ++nSvd)
      if(svds.singularValues()[nSvd] < CUTOFF_) break;

//  std::cout << "\t\t\tTruncating " << std::setw(4) << svds.singularValues().size() << " vectors to " << std::setw(4) << nSvd << std::endl;

    // No singular value is selected...
    if(nSvd == 0) continue;

    nSel += nSvd;

    qS.push_back(qS0[k]);

    lambda.push_back(std::vector<double>(nSvd));
    for(int kSel = 0; kSel < nSvd; ++kSel)
      lambda[nSym][kSel] = svds.singularValues()[kSel];

//DEBUG
//  std::cout << "\t\t\tblock size :: " << std::setw(4) << lambda[k].size() << " ";
//  for(int ksel = 0; ksel < nSvd; ++ksel) {
//    std::cout << std::setw(10) << std::scientific << std::setprecision(2) << lambda[k][ksel];
//    if(ksel % 10 == 9) std::cout << std::endl;
//  }
//  if(lambda[k].size() % 10 > 0) std::cout << std::endl;
//DEBUG

    if(iu >= 0 && prow != 0)
      ua(iu,nSym).reset(new local_matrix_type<double>(U.block(   0,   0,     prow,nSvd)));

    if(id >= 0 && prow != nrow)
      da(id,nSym).reset(new local_matrix_type<double>(U.block(prow,   0,nrow-prow,nSvd)));

    if(ju >= 0 && pcol != 0)
      ub(nSym,ju).reset(new local_matrix_type<double>(Vt.block(   0,   0,nSvd,     pcol)));

    if(jd >= 0 && pcol != ncol)
      db(nSym,jd).reset(new local_matrix_type<double>(Vt.block(   0,pcol,nSvd,ncol-pcol)));

    ++nSym;
  }

  aMps.matrix_u.resize(qR.size(),qS.size());
  aMps.matrix_d.resize(qR.size(),qS.size());
  for(size_t i = 0; i < qR.size(); ++i)
    for(size_t j = 0; j < qS.size(); ++j) {
      aMps.matrix_u(i,j) = ua(i,j);
      aMps.matrix_d(i,j) = da(i,j);
    }

  bMps.matrix_u.resize(qS.size(),qC.size());
  bMps.matrix_d.resize(qS.size(),qC.size());
  for(size_t i = 0; i < qS.size(); ++i)
    for(size_t j = 0; j < qC.size(); ++j) {
      bMps.matrix_u(i,j) = ub(i,j);
      bMps.matrix_d(i,j) = db(i,j);
    }

//std::cout << "\t\t\t" << std::setw(4) << nSel << " vectors are selected..." << std::endl;

}
