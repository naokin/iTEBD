#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>
#include <functional>

#include <btas/QSPARSE/QSTArray.h>
#include "SpinQuantum.h"

void Display (
  const btas::QSTArray<double,3,SpinQuantum>& X)
{
  size_t nrow = X.shape(0);
  size_t ncol = X.shape(2);

  btas::TArray<int,2> Xu(nrow,ncol); Xu = 0;
  btas::TArray<int,2> Xd(nrow,ncol); Xd = 0;

  for(auto iX = X.begin(); iX != X.end(); ++iX) {
    auto index = X.index(iX->first);
    if(index[1] == 0)
      Xu(index[0],index[2]) = 1;
    else
      Xd(index[0],index[2]) = 1;
  }

  for(size_t i = 0; i < nrow; ++i) {
    std::cout << "\t" << X.qshape(0)[i] << " :: ";
    for(size_t j = 0; j < ncol; ++j)
      std::cout << Xu(i,j) << " ";
    std::cout << "  ";
    for(size_t j = 0; j < ncol; ++j)
      std::cout << Xd(i,j) << " ";
    std::cout << std::endl;
  }
}

void Display (
  const btas::QSTArray<double,4,SpinQuantum>& X)
{
  size_t nrow = X.shape(0);
  size_t ncol = X.shape(3);

  btas::TArray<int,2> Xuu(nrow,ncol); Xuu = 0;
  btas::TArray<int,2> Xud(nrow,ncol); Xud = 0;
  btas::TArray<int,2> Xdu(nrow,ncol); Xdu = 0;
  btas::TArray<int,2> Xdd(nrow,ncol); Xdd = 0;

  for(auto iX = X.begin(); iX != X.end(); ++iX) {
    auto index = X.index(iX->first);
    if(index[1] == 0 && index[2] == 0)
      Xuu(index[0],index[3]) = 1;
    if(index[1] == 0 && index[2] == 1)
      Xud(index[0],index[3]) = 1;
    if(index[1] == 1 && index[2] == 0)
      Xdu(index[0],index[3]) = 1;
    if(index[1] == 1 && index[2] == 1)
      Xdd(index[0],index[3]) = 1;
  }

  for(size_t i = 0; i < nrow; ++i) {
    std::cout << "\t" << X.qshape(0)[i] << " :: ";
    for(size_t j = 0; j < ncol; ++j)
      std::cout << Xuu(i,j) << " ";
    std::cout << "  ";
    for(size_t j = 0; j < ncol; ++j)
      std::cout << Xud(i,j) << " ";
    std::cout << std::endl;
  }
  std::cout << std::endl;
  for(size_t i = 0; i < nrow; ++i) {
    std::cout << "\t" << X.qshape(0)[i] << " :: ";
    for(size_t j = 0; j < ncol; ++j)
      std::cout << Xdu(i,j) << " ";
    std::cout << "  ";
    for(size_t j = 0; j < ncol; ++j)
      std::cout << Xdd(i,j) << " ";
    std::cout << std::endl;
  }
}

void MPS_init (
        btas::QSTArray<double,3,SpinQuantum>& A, btas::STArray<double,1>& LA,
        btas::QSTArray<double,3,SpinQuantum>& B, btas::STArray<double,1>& LB,
        int M_spin)
{
  std::mt19937 rGen;
  std::uniform_real_distribution<double> dist(-1.0,1.0);

  btas::Qshapes<SpinQuantum> qh;
  qh.push_back(SpinQuantum(+1));
  qh.push_back(SpinQuantum(-1));

  btas::Qshapes<SpinQuantum> qA;
  for(int i = -M_spin; i <= M_spin; i += 2)
    qA.push_back(SpinQuantum(i));

  btas::Qshapes<SpinQuantum> qB;
  for(int i = -M_spin+1; i < M_spin; i += 2)
    qB.push_back(SpinQuantum(i));

  btas::Dshapes dh(qh.size(),1);

  btas::Dshapes dA(qA.size(),10);

  btas::Dshapes dB(qB.size(),10);

  LA.resize(btas::make_array(dA),1.0);
//A.resize(SpinQuantum(0),btas::make_array(qB,qh,-qA),btas::make_array(dB,dh,dA),std::bind(dist,rGen));
  A.resize(SpinQuantum(0),btas::make_array(qB,qh,-qA),btas::make_array(dB,dh,dA),1.0);

  LB.resize(btas::make_array(dB),1.0);
//B.resize(SpinQuantum(0),btas::make_array(qA,qh,-qB),btas::make_array(dA,dh,dB),std::bind(dist,rGen));
  B.resize(SpinQuantum(0),btas::make_array(qA,qh,-qB),btas::make_array(dA,dh,dB),1.0);
}

void fix_left_gauge (
  const btas::STArray<double,1>& G,
        btas::QSTArray<double,3,SpinQuantum>& X)
{
  btas::Dimm(G,X);
}

void fix_right_gauge (
  const btas::STArray<double,1>& G,
        btas::QSTArray<double,3,SpinQuantum>& X)
{
  btas::Dimm(X,G);
}

void fix_left_gauge_inverse (
  const btas::STArray<double,1>& G,
        btas::QSTArray<double,3,SpinQuantum>& X)
{
  btas::STArray<double,1> iG(G);
  for(auto it = iG.begin(); it != iG.end(); ++it)
    for(auto iit = it->second->begin(); iit != it->second->end(); ++iit) (*iit) = 1.0/(*iit);
  btas::Dimm(iG,X);
}

void fix_right_gauge_inverse (
  const btas::STArray<double,1>& G,
        btas::QSTArray<double,3,SpinQuantum>& X)
{
  btas::STArray<double,1> iG(G);
  for(auto it = iG.begin(); it != iG.end(); ++it)
    for(auto iit = it->second->begin(); iit != it->second->end(); ++iit) (*iit) = 1.0/(*iit);
  btas::Dimm(X,iG);
}

double propagate (
  const btas::QSTArray<double,4,SpinQuantum>& h,
        btas::QSTArray<double,3,SpinQuantum>& A, btas::STArray<double,1>& LA,
        btas::QSTArray<double,3,SpinQuantum>& B, btas::STArray<double,1>& LB,
        size_t Chi = 0)
{
  fix_right_gauge(LB,B);


  btas::QSTArray<double,4,SpinQuantum> Wfn;
  btas::Contract(1.0,A,btas::shape(2),B,btas::shape(0),1.0,Wfn);
  double WfnNorm2 = btas::Dotc(Wfn,Wfn);

//Display(Wfn);
  std::cout << "Wavefunction Square Norm = " << std::fixed << std::setw(16) << std::setprecision(12) << WfnNorm2 << std::endl;

  btas::QSTArray<double,4,SpinQuantum> Sgv;
  btas::Contract(1.0,Wfn,btas::shape(0,1,2,3),h,btas::shape(4,5,1,2),1.0,Sgv,btas::shape(0,4,5,3));
  double SgvNorm2 = btas::Dotc(Sgv,Sgv);

//Display(Sgv);
  std::cout << "Evolved wave Square Norm = " << std::fixed << std::setw(16) << std::setprecision(12) << SgvNorm2 << std::endl;

  btas::Gesvd(Sgv,LA,A,B,Chi);

  double Norm2 = btas::Dot(LA,LA);
  btas::Scal(1.0/sqrt(Norm2),LA);

  fix_left_gauge(LA,B);
  fix_right_gauge_inverse(LB,B);

  return -log(SgvNorm2)/WfnNorm2/2.0;
}

int main () {

  using std::cout;
  using std::endl;
  using std::setw;
  using std::fixed;

  double J  = 1.0;
  double Jz = 1.0;
  double Hz = 0.0;
  size_t Chi= 0;

  size_t Nt = 100;
  double dt = 0.3;

  // h(lb,rb,lk,rk)
  btas::Qshapes<SpinQuantum> qh;
  qh.push_back(SpinQuantum(+1));
  qh.push_back(SpinQuantum(-1));
  btas::QSTArray<double,4,SpinQuantum> h(SpinQuantum(0),btas::make_array(qh,qh,-qh,-qh));

  double expJz = exp(-0.25*Jz*dt);
  double expHz = exp(-Hz*dt);
  double coshJ = cosh(0.5*J*dt);
  double sinhJ = sinh(0.5*J*dt);

  btas::TArray<double,4> data_0000(1,1,1,1); data_0000 = expJz*expHz;
  btas::TArray<double,4> data_0101(1,1,1,1); data_0101 = coshJ/expJz;
  btas::TArray<double,4> data_0110(1,1,1,1); data_0110 =-sinhJ/expJz;
  btas::TArray<double,4> data_1111(1,1,1,1); data_1111 = expJz/expHz;

  h.insert(btas::shape(0,0,0,0),data_0000);
  h.insert(btas::shape(0,1,0,1),data_0101);
  h.insert(btas::shape(1,0,1,0),data_0101);
  h.insert(btas::shape(0,1,1,0),data_0110);
  h.insert(btas::shape(1,0,0,1),data_0110);
  h.insert(btas::shape(1,1,1,1),data_1111);

  btas::QSTArray<double,3,SpinQuantum> A,B;
  btas:: STArray<double,1> LA,LB;
  MPS_init(A,LA,B,LB,5);

  cout.precision(8);

  for(int i = 0; i < Nt; ++i) {

//  std::cout << "A->B :: MPS A :: " << std::endl;
//  Display(A);

//  std::cout << "A->B :: MPS B :: " << std::endl;
//  Display(B);

    double E_AB = propagate(h,A,LA,B,LB,Chi)/dt;
    cout << "\tStep[" << setw(4) << i << "] E[A-B]:: " << fixed << setw(12) << E_AB << endl;

    auto iLA = LA.begin();
    for(int i = 0; i < LA.size(); ++i, ++iLA) {
      if(i % 20 == 0) std::cout << "\t\tBlock size :: ";
      std::cout << std::setw(4) << iLA->second->size() << " ";
      if(i % 20 == 19) std::cout << std::endl;
    }
    if(LA.size() % 20 > 0) std::cout << std::endl;

//  std::cout << "B->A :: MPS A :: " << std::endl;
//  Display(A);

//  std::cout << "B->A :: MPS B :: " << std::endl;
//  Display(B);

    double E_BA = propagate(h,B,LB,A,LA,Chi)/dt;
    cout << "\tStep[" << setw(4) << i << "] E[B-A]:: " << fixed << setw(12) << E_BA << endl;

    auto iLB = LB.begin();
    for(int i = 0; i < LB.size(); ++i, ++iLB) {
      if(i % 20 == 0) std::cout << "\t\tBlock size :: ";
      std::cout << std::setw(4) << iLB->second->size() << " ";
      if(i % 20 == 19) std::cout << std::endl;
    }
    if(LB.size() % 20 > 0) std::cout << std::endl;

  }

  return 0;
}
