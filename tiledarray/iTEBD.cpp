#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include "MPS.h"
#include "MPS_init.h"
#include "imagEvolve.h"

/// iTEBD with symmetry sector constraints, a large-SVD-free algorithm
/// Note that size of symmetry sectors, blocks, and states are fixed during time-evolution
void iTEBD (
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

  std::vector<int> qA;
  std::vector<double> lambdaA;
  MPS<double> mpsA;

  std::vector<int> qB;
  std::vector<double> lambdaB;
  MPS<double> mpsB;

  // NOTE: wfn = lambdaB * mpsA * lambdaA * mpsB * lambdaB

  if(world.rank() == 0) std::cout << "\tInitializing MPS..." << std::endl;

  MPS_init(world,qA,lambdaA,mpsA,qB,lambdaB,mpsB,M_spin,M_state);

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
    std::cout << "\tStarting imaginary time-evolution :: T = " << std::fixed << std::setw(12) << std::setprecision(6) << nStep*dt
                                                  << ", dt = " << std::fixed << std::setw(12) << std::setprecision(6) << dt << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
  }

  for(size_t t = 0; t < nStep; ++t) {
    // exp(-ht) acting on A-B
    E = imagEvolve(world,qA,lambdaA,mpsA,qB,lambdaB,mpsB,J,Jz,Hz,dt,tole);

    if(world.rank() == 0 && t % print_freq == 0) {
      std::cout << "\t\tForward  step [" << std::setw(6) << t << "] :: " << std::fixed << std::setw(12) << std::setprecision(8) << E << std::endl;
      std::cout << "\t\t\tqA = "; for(size_t i = 0; i < qA.size(); ++i) std::cout << std::setw(4) << qA[i];
      std::cout << " [ " << lambdaA.size() << " ] " << std::endl;
      std::cout << "----------------------------------------------------------------" << std::endl;
    }

    world.gop.fence();

    // exp(-ht) acting on B-A
    E = imagEvolve(world,qB,lambdaB,mpsB,qA,lambdaA,mpsA,J,Jz,Hz,dt,tole);

    if(world.rank() == 0 && t % print_freq == 0) {
      std::cout << "\t\tBackward step [" << std::setw(6) << t << "] :: " << std::fixed << std::setw(12) << std::setprecision(8) << E << std::endl;
      std::cout << "\t\t\tqB = "; for(size_t i = 0; i < qB.size(); ++i) std::cout << std::setw(4) << qB[i];
      std::cout << " [ " << lambdaB.size() << " ] " << std::endl;
      std::cout << "----------------------------------------------------------------" << std::endl;
    }

    world.gop.fence();
  }
}

// MAIN
int main (int argc, char* argv[])
{
  madness::World& world = madness::initialize(argc,argv);

  // iTEBD (J, Jz, Hz, dt, "# step",
  //        "# initial quanta",
  //        "# initial states for each quantum",
  //        "tolerance of singular value")
  iTEBD(world,1.0,1.0,0.0,1.0,100,8,1,1.0e-6);

  madness::finalize();

  return 0;
}

