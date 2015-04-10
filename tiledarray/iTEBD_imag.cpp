#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include "MPS.h"
#include "MPS_init.h"
#include "imagEvolve.h"

void printQuanta (const std::vector<int>& q, const TiledArray::TiledRange1& t1) {
  std::cout << "\t\tSpin[Dim] = ";
  auto it = t1.begin();
  for(size_t i = 0; i < q.size(); ++i, ++it) std::cout << q[i] << "/2[" << (it->second-it->first) << "] ";
  std::cout << ":: total[" << t1.elements().second-t1.elements().first << "] " << std::endl;
}

/// iTEBD for imaginary-time evolution on spin-1/2 1D-Heisenberg model
void iTEBD_imag (
      madness::World& world,
      double J, double Jz, double Hz, double dt, size_t nStep, int M_spin, size_t M_state, double tole)
{
  if(world.rank() == 0) {
    std::cout << "\tJ  = " << std::fixed << std::setw(6) << std::setprecision(2) << J  << std::endl;
    std::cout << "\tJz = " << std::fixed << std::setw(6) << std::setprecision(2) << Jz << std::endl;
    std::cout << "\tHz = " << std::fixed << std::setw(6) << std::setprecision(2) << Hz << std::endl;

    std::cout << "\t# spin  = " << std::setw(8) << M_spin  << std::endl;
    std::cout << "\t# state = " << std::setw(8) << M_state << std::endl;
  }

  // initializing MPS

  std::vector<int> qA, qB;

  std::vector<double> lambdaA, lambdaB;

  MPS<double> mpsA, mpsB;

  // NOTE: wfn = lambdaB * mpsA * lambdaA * mpsB * lambdaB

  if(world.rank() == 0) std::cout << "\tInitializing MPS..." << std::endl;

  MPS_init(world,qA,lambdaA,mpsA,qB,lambdaB,mpsB); // perform initial wave as anti-ferro state

  if(world.rank() == 0) {
    std::cout << "\t\tqA = "; for(size_t i = 0; i < qA.size(); ++i) std::cout << std::setw(4) << qA[i];
    std::cout << " [ " << lambdaA.size() << " ] " << std::endl;
    std::cout << "\t\tqB = "; for(size_t i = 0; i < qB.size(); ++i) std::cout << std::setw(4) << qB[i];
    std::cout << " [ " << lambdaB.size() << " ] " << std::endl;
  }

  // imaginary time-evolution

  double E = 0.0;

  const size_t print_freq = 1;

  if(world.rank() == 0) {
    std::cout << "\tStarting imaginary time-evolution :: T = "
              << std::fixed << std::setw(12) << std::setprecision(6) << nStep*dt
              << ", dt = " << std::fixed << std::setw(12) << std::setprecision(6) << dt << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
  }

  for(size_t t = 0; t < nStep; ++t) {
    // exp(-ht) acting on A-B
    E = imagEvolve(world,qA,lambdaA,mpsA,qB,lambdaB,mpsB,J,Jz,Hz,dt,tole);

    if(world.rank() == 0 && t % print_freq == 0) {
      std::cout << "\tA-B step [" << std::setw(6) << t << "] :: "
                << std::fixed << std::setw(12) << std::setprecision(8) << E << std::endl;
      printQuanta(qA,mpsA.matrix_u.trange().data()[1]);
      std::cout << "----------------------------------------------------------------" << std::endl;
    }

    world.gop.fence();

    // exp(-ht) acting on B-A
    E = imagEvolve(world,qB,lambdaB,mpsB,qA,lambdaA,mpsA,J,Jz,Hz,dt,tole);

    if(world.rank() == 0 && t % print_freq == 0) {
      std::cout << "\tB-A step [" << std::setw(6) << t << "] :: "
                << std::fixed << std::setw(12) << std::setprecision(8) << E << std::endl;
      printQuanta(qB,mpsB.matrix_u.trange().data()[1]);
      std::cout << "----------------------------------------------------------------" << std::endl;
    }

    world.gop.fence();
  }
}

// MAIN
int main (int argc, char* argv[])
{
  madness::World& world = madness::initialize(argc,argv);

  // iTEBD (world, J, Jz, Hz, dt, "# step",
  //        "# initial quanta",
  //        "# initial states for each quantum",
  //        "tolerance of singular value")
  iTEBD_imag(world,1.0,1.0,0.0,0.3,100,8,1,1.0e-8);

  madness::finalize();

  return 0;
}

