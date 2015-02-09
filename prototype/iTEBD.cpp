#include <iostream>
#include <iomanip>

#include <cmath>

#include "MatrixFunctions.h"
#include "MPS.h"
#include "Wavefunction.h"
#include "GetWavefunction.h"
#include "SparseSVD.h"

void Display (const MPS<double>& X)
{
  size_t nrow = X.matrix_u.rows(); assert(X.matrix_d.rows() == nrow);
  size_t ncol = X.matrix_u.cols(); assert(X.matrix_d.cols() == ncol);

  for(size_t i = 0; i < nrow; ++i) {
    std::cout << "\t";
    for(size_t j = 0; j < ncol; ++j) {
      if(X.matrix_u(i,j))
        std::cout << "1 ";
      else
        std::cout << "0 ";
    }
    std::cout << "\t";
    for(size_t j = 0; j < ncol; ++j) {
      if(X.matrix_d(i,j))
        std::cout << "1 ";
      else
        std::cout << "0 ";
    }
    std::cout << std::endl;
  }
}

void Display (const Wavefunction<double>& X)
{
  size_t nrow = X.matrix_uu.rows();
  assert(X.matrix_ud.rows() == nrow);
  assert(X.matrix_du.rows() == nrow);
  assert(X.matrix_dd.rows() == nrow);
  size_t ncol = X.matrix_uu.cols();
  assert(X.matrix_ud.cols() == ncol);
  assert(X.matrix_du.cols() == ncol);
  assert(X.matrix_dd.cols() == ncol);

  for(size_t i = 0; i < nrow; ++i) {
    std::cout << "\t";
    for(size_t j = 0; j < ncol; ++j) {
      if(X.matrix_uu(i,j))
        std::cout << "1 ";
      else
        std::cout << "0 ";
    }
    std::cout << "\t";
    for(size_t j = 0; j < ncol; ++j) {
      if(X.matrix_ud(i,j))
        std::cout << "1 ";
      else
        std::cout << "0 ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  for(size_t i = 0; i < nrow; ++i) {
    std::cout << "\t";
    for(size_t j = 0; j < ncol; ++j) {
      if(X.matrix_du(i,j))
        std::cout << "1 ";
      else
        std::cout << "0 ";
    }
    std::cout << "\t";
    for(size_t j = 0; j < ncol; ++j) {
      if(X.matrix_dd(i,j))
        std::cout << "1 ";
      else
        std::cout << "0 ";
    }
    std::cout << std::endl;
  }
}

void MPS_init (
  std::vector<int>& aQ,
  std::vector<std::vector<double>>& aLambda,
  MPS<double>& aMps,
  std::vector<int>& bQ,
  std::vector<std::vector<double>>& bLambda,
  MPS<double>& bMps,
  int M_spin, size_t M_state)
{
  aQ.clear(); aQ.reserve(M_spin+1);
  bQ.clear(); bQ.reserve(M_spin+1);

  // define spin symmetry sector

  for(int k = -M_spin  ; k <= M_spin; k+=2) aQ.push_back(k);
  for(int k = -M_spin+1; k <  M_spin; k+=2) bQ.push_back(k);

  // init lambda

  aLambda.resize(aQ.size());
  for(int k = 0; k < aQ.size(); ++k) {
    aLambda[k].resize(M_state);
    std::fill(aLambda[k].begin(),aLambda[k].end(),1.0);
  }

  bLambda.resize(bQ.size());
  for(int k = 0; k < bQ.size(); ++k) {
    bLambda[k].resize(M_state);
    std::fill(bLambda[k].begin(),bLambda[k].end(),1.0);
  }

  // make matrix A

  aMps.matrix_u = matrix_type<double>::Constant(bQ.size(),aQ.size(),0);
  aMps.matrix_d = matrix_type<double>::Constant(bQ.size(),aQ.size(),0);

  for(int i = 0; i < bQ.size(); ++i) {
    for(int k = 0; k < aQ.size(); ++k) {
      if((bQ[i]+1) == aQ[k])
        aMps.matrix_u(i,k).reset(new local_matrix_type<double>(local_matrix_type<double>::Constant(M_state,M_state,1.0)));
//      aMps.matrix_u(i,k).reset(new local_matrix_type<double>(local_matrix_type<double>::Random(M_state,M_state)));
      if((bQ[i]-1) == aQ[k])
        aMps.matrix_d(i,k).reset(new local_matrix_type<double>(local_matrix_type<double>::Constant(M_state,M_state,1.0)));
//      aMps.matrix_d(i,k).reset(new local_matrix_type<double>(local_matrix_type<double>::Random(M_state,M_state)));
    }
  }

  // make matrix B

  bMps.matrix_u = matrix_type<double>::Constant(aQ.size(),bQ.size(),0);
  bMps.matrix_d = matrix_type<double>::Constant(aQ.size(),bQ.size(),0);

  for(int k = 0; k < aQ.size(); ++k) {
    for(int j = 0; j < bQ.size(); ++j) {
      if(aQ[k] == (bQ[j]-1))
        bMps.matrix_u(k,j).reset(new local_matrix_type<double>(local_matrix_type<double>::Constant(M_state,M_state,1.0)));
//      bMps.matrix_u(k,j).reset(new local_matrix_type<double>(local_matrix_type<double>::Random(M_state,M_state)));
      if(aQ[k] == (bQ[j]+1))
        bMps.matrix_d(k,j).reset(new local_matrix_type<double>(local_matrix_type<double>::Constant(M_state,M_state,1.0)));
//      bMps.matrix_d(k,j).reset(new local_matrix_type<double>(local_matrix_type<double>::Random(M_state,M_state)));
    }
  }
}

void l_gauge_fix (
  const std::vector<std::vector<double>>& g, MPS<double>& mps)
{
  for(size_t i = 0; i < mps.matrix_u.rows(); ++i) {
    for(size_t j = 0; j < mps.matrix_u.cols(); ++j) {
      if(!mps.matrix_u(i,j)) continue;
      local_matrix_type<double>& x = *mps.matrix_u(i,j);
      for(size_t ix = 0; ix < x.rows(); ++ix)
        for(size_t jx = 0; jx < x.cols(); ++jx) x(ix,jx) *= g[i][ix];
    }
  }

  for(size_t i = 0; i < mps.matrix_d.rows(); ++i) {
    for(size_t j = 0; j < mps.matrix_d.cols(); ++j) {
      if(!mps.matrix_d(i,j)) continue;
      local_matrix_type<double>& x = *mps.matrix_d(i,j);
      for(size_t ix = 0; ix < x.rows(); ++ix)
        for(size_t jx = 0; jx < x.cols(); ++jx) x(ix,jx) *= g[i][ix];
    }
  }
}

void r_gauge_fix (
  const std::vector<std::vector<double>>& g, MPS<double>& mps)
{
  for(size_t i = 0; i < mps.matrix_u.rows(); ++i) {
    for(size_t j = 0; j < mps.matrix_u.cols(); ++j) {
      if(!mps.matrix_u(i,j)) continue;
      local_matrix_type<double>& x = *mps.matrix_u(i,j);
      for(size_t ix = 0; ix < x.rows(); ++ix)
        for(size_t jx = 0; jx < x.cols(); ++jx) x(ix,jx) *= g[j][jx];
    }
  }

  for(size_t i = 0; i < mps.matrix_d.rows(); ++i) {
    for(size_t j = 0; j < mps.matrix_d.cols(); ++j) {
      if(!mps.matrix_d(i,j)) continue;
      local_matrix_type<double>& x = *mps.matrix_d(i,j);
      for(size_t ix = 0; ix < x.rows(); ++ix)
        for(size_t jx = 0; jx < x.cols(); ++jx) x(ix,jx) *= g[j][jx];
    }
  }
}

void l_gauge_fix_inverse (
  const std::vector<std::vector<double>>& g, MPS<double>& mps)
{
  for(size_t i = 0; i < mps.matrix_u.rows(); ++i) {
    for(size_t j = 0; j < mps.matrix_u.cols(); ++j) {
      if(!mps.matrix_u(i,j)) continue;
      local_matrix_type<double>& x = *mps.matrix_u(i,j);
      for(size_t ix = 0; ix < x.rows(); ++ix)
        for(size_t jx = 0; jx < x.cols(); ++jx) x(ix,jx) /= g[i][ix];
    }
  }

  for(size_t i = 0; i < mps.matrix_d.rows(); ++i) {
    for(size_t j = 0; j < mps.matrix_d.cols(); ++j) {
      if(!mps.matrix_d(i,j)) continue;
      local_matrix_type<double>& x = *mps.matrix_d(i,j);
      for(size_t ix = 0; ix < x.rows(); ++ix)
        for(size_t jx = 0; jx < x.cols(); ++jx) x(ix,jx) /= g[i][ix];
    }
  }
}

void r_gauge_fix_inverse (
  const std::vector<std::vector<double>>& g, MPS<double>& mps)
{
  for(size_t i = 0; i < mps.matrix_u.rows(); ++i) {
    for(size_t j = 0; j < mps.matrix_u.cols(); ++j) {
      if(!mps.matrix_u(i,j)) continue;
      local_matrix_type<double>& x = *mps.matrix_u(i,j);
      for(size_t ix = 0; ix < x.rows(); ++ix)
        for(size_t jx = 0; jx < x.cols(); ++jx) x(ix,jx) /= g[j][jx];
    }
  }

  for(size_t i = 0; i < mps.matrix_d.rows(); ++i) {
    for(size_t j = 0; j < mps.matrix_d.cols(); ++j) {
      if(!mps.matrix_d(i,j)) continue;
      local_matrix_type<double>& x = *mps.matrix_d(i,j);
      for(size_t ix = 0; ix < x.rows(); ++ix)
        for(size_t jx = 0; jx < x.cols(); ++jx) x(ix,jx) /= g[j][jx];
    }
  }
}

/// Make a Trotter step of imaginary time-evolution for the ground state search
/// DEF.: A = aMps, p = aLambda, B = bMps, q = bLambda
/// ALGO:
/// 1) A = q.a, B = p.b
/// 2) C = A.B.q = q.a.p.b.q = A'.p.B' by doing SVD
/// 3) normalize(p)
/// 4) A"= A'= q.a
/// 5) B"= p.B'.(1/q) = p.b
double imagEvolve (
  std::vector<int>& aQ,
  std::vector<std::vector<double>>& aLambda,
  MPS<double>& aMps,
  std::vector<int>& bQ,
  std::vector<std::vector<double>>& bLambda,
  MPS<double>& bMps,
  double J, double Jz, double Hz, double dt, double tole)
{
  r_gauge_fix(bLambda,bMps);

//std::cout << "\tContracting MPSs to form 2-ste wavefunction..." << std::endl;
  Wavefunction<double> wfn;
  GetWavefunction(aMps,bMps,wfn);

//std::cout << "\tPrinting Wavefunction :: " << std::endl;
//Display(wfn);
//std::cout << std::endl;

  double wfnNorm2 = 0.0;
  wfnNorm2 += SquareNorm(wfn.matrix_uu);
  wfnNorm2 += SquareNorm(wfn.matrix_ud);
  wfnNorm2 += SquareNorm(wfn.matrix_du);
  wfnNorm2 += SquareNorm(wfn.matrix_dd);
//std::cout << "\t\tSquare norm [wfn] = " << std::fixed << std::setw(24) << std::setprecision(20) << wfnNorm2 << std::endl;

  // Compute exp(-h*dt)*wfn

  // Nearest neighbour propagator
  //         +1                                                     -1
  //         +1                         -1                          +1                          -1
  // ------+--------------------------------------------------------------------------------------------------------------
  // +1 +1 |  exp(-Jz*dt/4)*exp(-Hz*dt)  0                           0                           0
  //    -1 |  0                          exp(+Jz*dt/4)*cosh(J*dt/2) -exp(+Jz*dt/4)*sinh(J*dt/2)  0
  // -1 +1 |  0                         -exp(+Jz*dt/4)*sinh(J*dt/2)  exp(+Jz*dt/4)*cosh(J*dt/2)  0
  //    -1 |  0                          0                           0                           exp(-Jz*dt/4)*exp(+Hz*dt)

//std::cout << "\tEvolving wavefunction..." << std::endl;

  double expJz = exp(-0.25*Jz*dt);
  double expHz = exp(-Hz*dt);
  double coshJ = cosh(0.5*J*dt);
  double sinhJ = sinh(0.5*J*dt);

  {
    matrix_type<double> ud_tmp; DeepCopy(wfn.matrix_ud,ud_tmp);
    matrix_type<double> du_tmp; DeepCopy(wfn.matrix_du,du_tmp);

    Scale(expJz*expHz,wfn.matrix_uu);
    Scale(coshJ/expJz,wfn.matrix_ud); ScaledAdd(-sinhJ/expJz,du_tmp,wfn.matrix_ud);
    Scale(coshJ/expJz,wfn.matrix_du); ScaledAdd(-sinhJ/expJz,ud_tmp,wfn.matrix_du);
    Scale(expJz/expHz,wfn.matrix_dd);
  }

//std::cout << "\tPrinting Evolved-Wave :: " << std::endl;
//Display(wfn);
//std::cout << std::endl;

  double sgvNorm2 = 0.0;
  sgvNorm2 += SquareNorm(wfn.matrix_uu);
  sgvNorm2 += SquareNorm(wfn.matrix_ud);
  sgvNorm2 += SquareNorm(wfn.matrix_du);
  sgvNorm2 += SquareNorm(wfn.matrix_dd);
//std::cout << "\t\tSquare norm [sgv] = " << std::fixed << std::setw(24) << std::setprecision(20) << sgvNorm2 << std::endl;

  matrix_type<double>& uu_ref = wfn.matrix_uu;
  matrix_type<double>& ud_ref = wfn.matrix_ud;
  matrix_type<double>& du_ref = wfn.matrix_du;
  matrix_type<double>& dd_ref = wfn.matrix_dd;

  // SVD : This is too complicated...

//std::cout << "\tDoing SVD to canonicalize MPSs..." << std::endl;
  SparseSVD(bQ,bQ,wfn,aQ,aLambda,aMps,bMps,tole);

//MPS<double> aMpsTmp;
//DeepCopy(aMps.matrix_u,aMpsTmp.matrix_u);
//DeepCopy(aMps.matrix_d,aMpsTmp.matrix_d);
//r_gauge_fix(aLambda,aMpsTmp);
//Wavefunction<double> wfnTmp;
//GetWavefunction(aMpsTmp,bMps,wfnTmp);
//double sgvNorm2Chk = 0.0;
//sgvNorm2Chk += DotProduct(wfn.matrix_uu,wfnTmp.matrix_uu);
//sgvNorm2Chk += DotProduct(wfn.matrix_ud,wfnTmp.matrix_ud);
//sgvNorm2Chk += DotProduct(wfn.matrix_du,wfnTmp.matrix_du);
//sgvNorm2Chk += DotProduct(wfn.matrix_dd,wfnTmp.matrix_dd);
//std::cout << "\t\tCheck  norm [sgv] = " << std::fixed << std::setw(24) << std::setprecision(20) << sgvNorm2Chk << std::endl;

  double aNorm2 = 0.0;
  for(int k = 0; k < aLambda.size(); ++k)
    for(int kx = 0; kx < aLambda[k].size(); ++kx)
      aNorm2 += aLambda[k][kx]*aLambda[k][kx];
//std::cout << "\t\tDEBUG ::   aNorm2              = " << std::fixed << std::setw(24) << std::setprecision(20) << aNorm2 << std::endl;

  double aNorm  = sqrt(aNorm2);
  for(int k = 0; k < aLambda.size(); ++k)
    for(int kx = 0; kx < aLambda[k].size(); ++kx)
      aLambda[k][kx] /= aNorm;

  l_gauge_fix        (aLambda,bMps);
  r_gauge_fix_inverse(bLambda,bMps);

  return -log(sgvNorm2)/wfnNorm2/dt/2.0;
}

/// iTEBD with symmetry sector constraints, a large-SVD-free algorithm
/// Note that size of symmetry sectors, blocks, and states are fixed during time-evolution
void iTEBD (
  double J, double Jz, double Hz, double dt, size_t nStep,
  int M_spin, size_t M_state, double tole)
{
  std::cout << "\tJ  = " << std::fixed << std::setw(6) << std::setprecision(2) << J  << std::endl;
  std::cout << "\tJz = " << std::fixed << std::setw(6) << std::setprecision(2) << Jz << std::endl;
  std::cout << "\tHz = " << std::fixed << std::setw(6) << std::setprecision(2) << Hz << std::endl;

  std::cout << "\t# spin  = " << std::setw(8) << M_spin  << std::endl;
  std::cout << "\t# state = " << std::setw(8) << M_state << std::endl;

  // initializing MPS

  std::vector<int> aQ;
  std::vector<std::vector<double>> aLambda;
  MPS<double> aMps;

  std::vector<int> bQ;
  std::vector<std::vector<double>> bLambda;
  MPS<double> bMps;

  // NOTE: wfn = bLambda * aMps * aLambda * bMps * bLambda
  std::cout << "\tInitializing MPS..." << std::endl;
  MPS_init(aQ,aLambda,aMps,bQ,bLambda,bMps,M_spin,M_state);

  std::cout << "\t\taQ = "; for(size_t i = 0; i < aQ.size(); ++i) std::cout << std::setw(4) << aQ[i]; std::cout << std::endl;

//std::cout << "\tPrinting MPS A :: " << std::endl;
//Display(aMps);
//std::cout << std::endl;

  std::cout << "\t\tbQ = "; for(size_t i = 0; i < bQ.size(); ++i) std::cout << std::setw(4) << bQ[i]; std::cout << std::endl;

//std::cout << "\tPrinting MPS B :: " << std::endl;
//Display(bMps);
//std::cout << std::endl;

  // imaginary time-evolution

  double E = 0.0;

  std::cout << "\tStarting imaginary time-evolution :: T = " << std::fixed << std::setw(12) << std::setprecision(6) << nStep*dt
                                                << ", dt = " << std::fixed << std::setw(12) << std::setprecision(6) << dt << std::endl;

  std::cout << "----------------------------------------------------------------" << std::endl;

  for(size_t t = 0; t < nStep; ++t) {
    // exp(-ht) acting on A-B
    E = imagEvolve(aQ,aLambda,aMps,bQ,bLambda,bMps,J,Jz,Hz,dt,tole);

    if(t % 10 == 0) {

    std::cout << "\t\tForward  step [" << std::setw(6) << t << "] :: " << std::fixed << std::setw(12) << std::setprecision(8) << E << std::endl;

    std::cout << "\t\t\taQ = "; for(size_t i = 0; i < aQ.size(); ++i) std::cout << std::setw(4) << aQ[i]; std::cout << std::endl;

    for(int i = 0; i < aLambda.size(); ++i) {
      if(i % 20 == 0) std::cout << "\t\t\tblock size :: ";
      std::cout << std::setw(4) << aLambda[i].size() << " ";
      if(i % 20 == 19) std::cout << std::endl;
    }
    if(aLambda.size() % 20 > 0) std::cout << std::endl;

//  std::cout << "\tPrinting MPS A :: " << std::endl;
//  Display(aMps);
//  std::cout << std::endl;

//  std::cout << "\tPrinting MPS B :: " << std::endl;
//  Display(bMps);
//  std::cout << std::endl;

    std::cout << "----------------------------------------------------------------" << std::endl;

    }

    // exp(-ht) acting on B-A
    E = imagEvolve(bQ,bLambda,bMps,aQ,aLambda,aMps,J,Jz,Hz,dt,tole);

    if(t % 10 == 0) {

    std::cout << "\t\tBackward step [" << std::setw(6) << t << "] :: " << std::fixed << std::setw(12) << std::setprecision(8) << E << std::endl;

    std::cout << "\t\t\tbQ = "; for(size_t i = 0; i < bQ.size(); ++i) std::cout << std::setw(4) << bQ[i]; std::cout << std::endl;

    for(int i = 0; i < bLambda.size(); ++i) {
      if(i % 20 == 0) std::cout << "\t\t\tblock size :: ";
      std::cout << std::setw(4) << bLambda[i].size() << " ";
      if(i % 20 == 19) std::cout << std::endl;
    }
    if(bLambda.size() % 20 > 0) std::cout << std::endl;

//  std::cout << "\tPrinting MPS A :: " << std::endl;
//  Display(aMps);
//  std::cout << std::endl;

//  std::cout << "\tPrinting MPS B :: " << std::endl;
//  Display(bMps);
//  std::cout << std::endl;

    std::cout << "----------------------------------------------------------------" << std::endl;

    }

  }
}

// MAIN
int main ()
{
  // iTEBD (J, Jz, Hz, dt, "# step",
  //        "# initial quanta",
  //        "# initial states for each quantum",
  //        "tolerance of singular value")
  iTEBD(1.0,1.0,0.0,1.0,10000,8,1,1.0e-6);

  return 0;
}

