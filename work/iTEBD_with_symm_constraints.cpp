#include <iostream>
#include <iomanip>

#include <cmath>

#include "iTEBD_with_symm_constraints.h"
#include "MPS.hpp"

namespace TA = TiledArray;

void iTEBD_with_symm_constraints
(
  double J, ///< J  coupling constant in 1D-Heisenberg spin hamiltonian
  double Jz,///< Jz coupling constant ...
  double Hz,///< external magnetic field
  double dt,///< step size of time-evolution
  size_t M_sector, ///< Number of symmetry sectors
  size_t M_block, ///< Number of blocks for each symmetry sector
  size_t M_state ///< Number of dense states for each block
)
{
  // iTEBD with symmetry sector constraints, a large-SVD-free algorithm
  // Size of symmetry sectors, blocks, and states are fixed during time-evolution

  // Structure of MPS A-matrix
  // E.g.)
  // M_sector = 4 (must be even No.)
  // M_block = 3 (arbitral)
  // M_state = dim.(NZ)
  //----+----+--------------+--------------+--------------+--------------+
  //    |    |      +3      |      +1      |      -1      |      -3      |
  //----+----+----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    | NZ |    |    |    |    |    |    |    |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    | +2 |    | NZ |    |    |    |    |    |    |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    | NZ |    |    |    |    |    |    |    |    |    |
  //    +----+----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    | NZ |    |    |    |    |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    | +0 |    |    |    |    | NZ |    |    |    |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    |    |    | NZ |    |    |    |    |    |    |
  // +1 +----+----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    | NZ |    |    |    |    |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    | -0 |    |    |    |    | NZ |    |    |    |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    |    |    | NZ |    |    |    |    |    |    |
  //    +----+----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    |    |    |    | NZ |    |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    | -2 |    |    |    |    |    |    |    | NZ |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    |    |    |    |    |    | NZ |    |    |    |
  //----+----+----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    | NZ |    |    |    |    |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    | +2 |    |    |    |    | NZ |    |    |    |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    |    |    | NZ |    |    |    |    |    |    |
  //    +----+----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    |    |    |    | NZ |    |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    | +0 |    |    |    |    |    |    |    | NZ |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    |    |    |    |    |    | NZ |    |    |    |
  // -1 +----+----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    |    |    |    | NZ |    |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    | -0 |    |    |    |    |    |    |    | NZ |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    |    |    |    |    |    | NZ |    |    |    |
  //    +----+----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    |    |    |    |    |    |    | NZ |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    | -2 |    |    |    |    |    |    |    |    |    |    | NZ |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    |    |    |    |    |    |    |    |    | NZ |
  //----+----+----+----+----+----+----+----+----+----+----+----+----+----+

  assert(M_sector%2 == 0);

  size_t nRange = M_sector*M_block;
  size_t nRangeHf = nRange/2;

  // TA:: Setup

  // TA:: Construct tile boundary vector
  std::vector<size_t> tile_boundaries(nRange+1);
  for(size_t i = 0; i <= nRange; ++i) {
    tile_boundaries[i] = i*M_state;
  }
  // TA:: Construct a set of 1D TiledRanges
  std::vector<TA::TiledRange1>
    ranges(2,TA::TiledRange1(tile_boundaries.begin(),tile_boundaries.end()));
  // TA:: Construct the 2D TiledRange
  TA::TiledRange trange(ranges.begin(),ranges.end());

  // Random number generator

  std::mt19937 randGen;
  std::uniform_real_distribution<double> realDist(-1.0,1.0);

  // Create A-matrix

  MPS<double> aMps;

  {
    // Create shape of spin-up matrix of A-matrix

    std::vector<size_t> shape_u;
    for(size_t i = 0; i < nRangeHf; ++i) {
      shape_u.push_back(i*nRange+i);
    }
    for(size_t i = nRangeHf; i < nRange; ++i) {
      shape_u.push_back(i*nRange+i-M_block);
    }

    aMps.matrix_u = MPS<double>::matrix_type(world,trange,shape_u.begin(),shape_u.end());

    for(MPS<double>::matrix_type::iterator it = aMps.matrix_u.begin(); it != aMps.matrix_u.end(); ++it) {
      // Construct a local_data
      MPS<double>::local_matrix_type local_data(aMps.matrix_u.trange().make_tile_range(it.ordinal()));
      // Generate random elements
      std::generate(local_data.begin(),local_data.end(),std::bind(realDist,randGen));
      // Insert the local_data to the array
      *it = local_data;
    }

    // Create shape of spin-down matrix of A-matrix

    std::vector<size_t> shape_d;
    for(size_t i = 0; i < nRangeHf; ++i) {
      shape_d.push_back(i*nRange+i+M_block);
    }
    for(size_t i = nRangeHf; i < nRange; ++i) {
      shape_d.push_back(i*nRange+i);
    }

    aMps.matrix_d = MPS<double>::matrix_type(world,trange,shape_d.begin(),shape_d.end());

    for(MPS<double>::matrix_type::iterator it = aMps.matrix_d.begin(); it != aMps.matrix_d.end(); ++it) {
      // Construct a local_data
      MPS<double>::local_matrix_type local_data(aMps.matrix_d.trange().make_tile_range(it.ordinal()));
      // Generate random elements
      std::generate(local_data.begin(),local_data.end(),std::bind(realDist,randGen));
      // Insert the local_data to the array
      *it = local_data;
    }
  }

  // Create B-matrix

  MPS<double> bMps;

  {
    // Create shape of spin-up matrix of B-matrix

    std::vector<size_t> shape_u;
    for(size_t i = 0; i < nRangeHf; ++i) {
      shape_u.push_back(i*nRange+M_block*nRange+i);
    }
    for(size_t i = nRangeHf; i < nRange; ++i) {
      shape_u.push_back(i*nRange+i);
    }

    bMps.matrix_u = MPS<double>::matrix_type(world,trange,shape_u.begin(),shape_u.end());

    for(MPS<double>::matrix_type::iterator it = bMps.matrix_u.begin(); it != bMps.matrix_u.end(); ++it) {
      // Construct a local_data
      MPS<double>::local_matrix_type local_data(bMps.matrix_u.trange().make_tile_range(it.ordinal()));
      // Generate random elements
      std::generate(local_data.begin(),local_data.end(),std::bind(realDist,randGen));
      // Insert the local_data to the array
      *it = local_data;
    }

    // Create shape of spin-down matrix of B-matrix

    std::vector<size_t> shape_d;
    for(size_t i = 0; i < nRangeHf; ++i) {
      shape_d.push_back(i*nRange+i);
    }
    for(size_t i = nRangeHf; i < nRange; ++i) {
      shape_d.push_back(i*nRange-M_block*nRange+i);
    }

    bMps.matrix_d = MPS<double>::matrix_type(world,trange,shape_d.begin(),shape_d.end());

    for(MPS<double>::matrix_type::iterator it = bMps.matrix_d.begin(); it != bMps.matrix_d.end(); ++it) {
      // Construct a local_data
      MPS<double>::local_matrix_type local_data(bMps.matrix_d.trange().make_tile_range(it.ordinal()));
      // Generate random elements
      std::generate(local_data.begin(),local_data.end(),std::bind(realDist,randGen));
      // Insert the local_data to the array
      *it = local_data;
    }
  }

  world.gop.fence();

  // Imaginary time-evolution

  // Gauge
  std::vector<double> aLambda(nRange*M_state,1.0);
  std::vector<double> bLambda(nRange*M_state,1.0);
  for(size_t t = 0; t < nStep; ++t) {
    // exp(-ht) acting on A-B
    imagEvolve(world,1,aMps,aLambda,bMps,bLambda,J,Jz,Hz,dt,M_sector,M_block,M_state);
    world.gop.fence();
    // exp(-ht) acting on B-A
    imagEvolve(world,0,bMps,bLambda,aMps,aLambda,J,Jz,Hz,dt,M_sector,M_block,M_state);
    world.gop.fence();
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
void imagEvolve
(
  madness::World& world,
  bool isForward,
  MPS<double>& aMps,
  std::vector<double>& aLambda,
  MPS<double>& bMps,
  std::vector<double>& bLambda,
  double J, ///< J  coupling constant in 1D-Heisenberg spin hamiltonian
  double Jz,///< Jz coupling constant ...
  double Hz,///< external magnetic field
  double dt,///< step size of time-evolution
  size_t M_sector, ///< Number of symmetry sectors
  size_t M_block, ///< Number of blocks for each symmetry sector
  size_t M_state ///< Number of dense states for each block
)
{
  // Multiplying gauge matrix to the bMps
  for(MPS<double>::matrix_type::iterator it = bMps.matrix_u.begin(); it != bMps.matrix_u.end(); ++it) {
    // Get a tile
    MPS<double>::local_matrix_type local_data(bMps.matrix_u.trange().make_tile_range(it.ordinal()));
    // Get a range of column
    size_t start  = local_data.range().start ()[1];
    size_t finish = local_data.range().finish()[1];
    // Multiply gauge matrix
    for(size_t i = 0; i < local_data.size();) {
      for(size_t j = start; j < finish; ++i, ++j)
        local_data[i] *= bLambda[j];
    }
    // Insert back the data
    *it = local_data;
  }
  for(MPS<double>::matrix_type::iterator it = bMps.matrix_d.begin(); it != bMps.matrix_d.end(); ++it) {
    // Get a tile
    MPS<double>::local_matrix_type local_data(bMps.matrix_d.trange().make_tile_range(it.ordinal()));
    // Get a range of column
    size_t start  = local_data.range().start ()[1];
    size_t finish = local_data.range().finish()[1];
    // Multiply gauge matrix
    for(size_t i = 0; i < local_data.size();) {
      for(size_t j = start; j < finish; ++i, ++j)
        local_data[i] *= bLambda[j];
    }
    // Insert back the data
    *it = local_data;
  }

  Wavefunction<double> wfn;

  // Compute aMps*bMps
  wfn.matrix_uu("l,r") = aMps.matrix_u("l,s")*bMps.matrix_u("s,r");
  wfn.matrix_ud("l,r") = aMps.matrix_u("l,s")*bMps.matrix_d("s,r");
  wfn.matrix_du("l,r") = aMps.matrix_d("l,s")*bMps.matrix_u("s,r");
  wfn.matrix_dd("l,r") = aMps.matrix_d("l,s")*bMps.matrix_d("s,r");

  // Compute exp(-h*dt)*wfn

  // Nearest neighbour propagator
  //         +1                                                     -1
  //         +1                         -1                          +1                          -1
  // ------+--------------------------------------------------------------------------------------------------------------
  // +1 +1 |  exp(-Jz*dt/4)*exp(-Hz*dt)  0                           0                           0
  //    -1 |  0                          exp(+Jz*dt/4)*cosh(J*dt/2) -exp(+Jz*dt/4)*sinh(J*dt/2)  0
  // -1 +1 |  0                         -exp(+Jz*dt/4)*sinh(J*dt/2)  exp(+Jz*dt/4)*cosh(J*dt/2)  0
  //    -1 |  0                          0                           0                           exp(-Jz*dt/4)*exp(+Hz*dt)

  double expJz = exp(-0.25*Jz*dt);
  double expHz = exp(-Hz*dt);
  double coshJ = cosh(0.5*J*dt);
  double sinhJ = sinh(0.5*J*dt);

  // Wavefunction<double>::matrix_type -- this is a type of block-sparse matrix

  {
    Wavefunction<double>::matrix_type ud_tmp(wfn.matrix_ud);
    Wavefunction<double>::matrix_type du_tmp(wfn.matrix_du);

    wfn.matrix_uu("l,r") = (expJz*expHz)*wfn.matrix_uu("l,r");
    wfn.matrix_ud("l,r") = (coshJ/expJz)*ud_tmp("l,r")-(sinhJ/expJz)*du_tmp("l,r");
    wfn.matrix_du("l,r") = (coshJ/expJz)*du_tmp("l,r")-(sinhJ/expJz)*ud_tmp("l,r");
    wfn.matrix_dd("l,r") = (expJz/expHz)*wfn.matrix_dd("l,r");
  }

  Wavefunction<double>::matrix_type& uu_ref = wfn.matrix_uu;
  Wavefunction<double>::matrix_type& ud_ref = wfn.matrix_ud;
  Wavefunction<double>::matrix_type& du_ref = wfn.matrix_du;
  Wavefunction<double>::matrix_type& dd_ref = wfn.matrix_dd;

  // SVD : This is too complicated...

#ifdef _BOOST_MPI_PARALLELISM
  size_t iproc = world.rank();
  size_t nproc = world.size();
#else
  size_t iproc = 0;
  size_t nproc = 1;
#endif

  if(isForward) {
    // There's 6 patterns
    size_t p0 = 0;
    size_t p1 = M_block;
    size_t p2 = M_sector*M_block/2-M_block;
    size_t p3 = p2+M_block;
    size_t p4 = p3+M_block;
    size_t p5 = p4+p2-M_block;
    size_t p6 = p5+M_block;
    // These loops can be parallelized...
    // Case 1
    for(size_t i = p0; i < p1; ++i) {
      if(i % nproc != iproc) continue;

      Eigen::Matrix<double> X(  M_state,  M_state); X.fill(0.0);
      X = ud_ref(i,i);

      // Canonicalizing M -> M
      Eigen::JacobiSVD<Matrix<double>> svds(X, Eigen::ComputeThinU | Eigen::ComputeThinV);

      Eigen::Matrix<double> uBuff = svds.matrixU();
      Eigen::Matrix<double> vBuff = svds.matrixV().transpose();

      aMps.matrix_u(i,i) = uBuff;
      bMps.matrix_d(i,i) = vBuff;
    }
    // Case 2
    for(size_t i = p1; i < p2; ++i) {
      if(i % nproc != iproc) continue;

      Eigen::Matrix<double> X(2*M_state,2*M_state); X.fill(0.0);
      X.block(      0,      0,M_state,M_state) = uu_ref(i        ,i-M_block);
      X.block(      0,M_state,M_state,M_state) = ud_ref(i        ,i        );
      X.block(M_state,      0,M_state,M_state) = du_ref(i-M_block,i-M_block);
      X.block(M_state,M_state,M_state,M_state) = dd_ref(i-M_block,i        );

      // Truncating 2M -> M
      Eigen::JacobiSVD<Matrix<double>> svds(X, Eigen::ComputeThinU | Eigen::ComputeThinV);

      Eigen::Matrix<double> uBuff = svds.matrixU().block(0,0,2*M_state,M_state);
      Eigen::Matrix<double> vBuff = svds.matrixV().block(0,0,2*M_state,M_state).transpose();

      aMps.matrix_u(i        ,i        ) = uBuff.block(      0,      0,M_state,M_state);
      aMps.matrix_d(i-M_block,i        ) = uBuff.block(M_state,      0,M_state,M_state);

      bMps.matrix_u(i        ,i-M_block) = vBuff.block(      0,      0,M_state,M_state);
      bMps.matrix_d(i        ,i        ) = vBuff.block(      0,M_state,M_state,M_state);
    }
    // Case 3
    for(size_t i = p2; i < p3; ++i) {
      if(i % nproc != iproc) continue;

      Eigen::Matrix<double> X(3*M_state,3*M_state); X.fill(0.0);
      X.block(        0,        0,M_state,M_state) = uu_ref(i        ,i-M_block);
      X.block(        0,  M_state,M_state,M_state) = ud_ref(i        ,i        );
      X.block(        0,2*M_state,M_state,M_state) = ud_ref(i        ,i+M_block);
      X.block(  M_state,        0,M_state,M_state) = uu_ref(i+M_block,i-M_block);
      X.block(  M_state,  M_state,M_state,M_state) = ud_ref(i+M_block,i        );
      X.block(  M_state,2*M_state,M_state,M_state) = ud_ref(i+M_block,i+M_block);
      X.block(2*M_state,        0,M_state,M_state) = du_ref(i-M_block,i-M_block);
      X.block(2*M_state,  M_state,M_state,M_state) = dd_ref(i-M_block,i        );
      X.block(2*M_state,2*M_state,M_state,M_state) = dd_ref(i-M_block,i+M_block);

      // Truncating 3M -> M
      Eigen::JacobiSVD<Matrix<double>> svds(X, Eigen::ComputeThinU | Eigen::ComputeThinV);

      Eigen::Matrix<double> uBuff = svds.matrixU().block(0,0,3*M_state,M_state);
      Eigen::Matrix<double> vBuff = svds.matrixV().block(0,0,3*M_state,M_state).transpose();

      aMps.matrix_u(i        ,i        ) = uBuff.block(        0,      0,M_state,M_state);
      aMps.matrix_u(i+M_block,i        ) = uBuff.block(  M_state,      0,M_state,M_state);
      aMps.matrix_d(i-M_block,i        ) = uBuff.block(2*M_state,      0,M_state,M_state);

      bMps.matrix_u(i        ,i-M_block) = vBuff.block(      0,        0,M_state,M_state);
      bMps.matrix_d(i        ,i        ) = vBuff.block(      0,  M_state,M_state,M_state);
      bMps.matrix_d(i        ,i+M_block) = vBuff.block(      0,2*M_state,M_state,M_state);
    }
    // Case 4
    for(size_t i = p3; i < p4; ++i) {
      if(i % nproc != iproc) continue;

      Eigen::Matrix<double> X(3*M_state,3*M_state); X.fill(0.0);
      X.block(        0,        0,M_state,M_state) = uu_ref(i+M_block,i-M_block);
      X.block(        0,  M_state,M_state,M_state) = uu_ref(i+M_block,i        );
      X.block(        0,2*M_state,M_state,M_state) = ud_ref(i+M_block,i+M_block);
      X.block(  M_state,        0,M_state,M_state) = du_ref(i-M_block,i-M_block);
      X.block(  M_state,  M_state,M_state,M_state) = du_ref(i-M_block,i        );
      X.block(  M_state,2*M_state,M_state,M_state) = dd_ref(i-M_block,i+M_block);
      X.block(2*M_state,        0,M_state,M_state) = du_ref(i        ,i-M_block);
      X.block(2*M_state,  M_state,M_state,M_state) = du_ref(i        ,i        );
      X.block(2*M_state,2*M_state,M_state,M_state) = dd_ref(i        ,i+M_block);

      // Truncating 3M -> M
      Eigen::JacobiSVD<Matrix<double>> svds(X, Eigen::ComputeThinU | Eigen::ComputeThinV);

      Eigen::Matrix<double> uBuff = svds.matrixU().block(0,0,3*M_state,M_state);
      Eigen::Matrix<double> vBuff = svds.matrixV().block(0,0,3*M_state,M_state).transpose();

      aMps.matrix_u(i+M_block,i        ) = uBuff.block(        0,      0,M_state,M_state);
      aMps.matrix_d(i-M_block,i        ) = uBuff.block(  M_state,      0,M_state,M_state);
      aMps.matrix_d(i        ,i        ) = uBuff.block(2*M_state,      0,M_state,M_state);

      bMps.matrix_u(i        ,i-M_block) = vBuff.block(      0,        0,M_state,M_state);
      bMps.matrix_u(i        ,i        ) = vBuff.block(      0,  M_state,M_state,M_state);
      bMps.matrix_d(i        ,i+M_block) = vBuff.block(      0,2*M_state,M_state,M_state);
    }
    // Case 5
    for(size_t i = p4; i < p5; ++i) {
      if(i % nproc != iproc) continue;

      Eigen::Matrix<double> X(2*M_state,2*M_state); X.fill(0.0);
      X.block(      0,      0,M_state,M_state) = uu_ref(i+M_block,i        );
      X.block(      0,M_state,M_state,M_state) = ud_ref(i+M_block,i+M_block);
      X.block(M_state,      0,M_state,M_state) = du_ref(i        ,i        );
      X.block(M_state,M_state,M_state,M_state) = dd_ref(i        ,i+M_block);

      // Truncating 2M -> M
      Eigen::JacobiSVD<Matrix<double>> svds(X, Eigen::ComputeThinU | Eigen::ComputeThinV);

      Eigen::Matrix<double> uBuff = svds.matrixU().block(0,0,2*M_state,M_state);
      Eigen::Matrix<double> vBuff = svds.matrixV().block(0,0,2*M_state,M_state).transpose();

      aMps.matrix_u(i+M_block,i        ) = uBuff.block(      0,      0,M_state,M_state);
      aMps.matrix_d(i        ,i        ) = uBuff.block(M_state,      0,M_state,M_state);

      bMps.matrix_u(i        ,i        ) = vBuff.block(      0,      0,M_state,M_state);
      bMps.matrix_d(i        ,i+M_block) = vBuff.block(      0,M_state,M_state,M_state);
    }
    // Case 6
    for(size_t i = p5; i < p6; ++i) {
      if(i % nproc != iproc) continue;

      Eigen::Matrix<double> X(M_state,M_state); X.fill(0.0);
      X = du_ref(i,i);

      // Canonicalizing M -> M
      Eigen::JacobiSVD<Matrix<double>> svds(X, Eigen::ComputeThinU | Eigen::ComputeThinV);

      Eigen::Matrix<double> uBuff = svds.matrixU();
      Eigen::Matrix<double> vBuff = svds.matrixV().transpose();

      aMps.matrix_d(i,i) = uBuff;
      bMps.matrix_u(i,i) = vBuff;
    }
  }
  else { // !isForward
    // There's 3 patterns
    size_t p0 = 0;
    size_t p1 = M_sector*M_block/2-M_block;
    size_t p2 = p1+M_block;
    size_t p3 = p2+p1;
    // These loops can be parallelized...
    // Case 1
    for(size_t i = p0; i < p1; ++i) {
      if(i % nproc != iproc) continue;
      Eigen::Matrix<double> X(2*M_state,2*M_state); X.fill(0.0);
      X.block(      0,      0,M_state,M_state) = uu_ref(i+M_block,i        );
      X.block(      0,M_state,M_state,M_state) = ud_ref(i+M_block,i+M_block);
      X.block(M_state,      0,M_state,M_state) = du_ref(i        ,i        );
      X.block(M_state,M_state,M_state,M_state) = dd_ref(i        ,i+M_block);

      // Truncating 2M -> M
      Eigen::JacobiSVD<Matrix<double>> svds(X, Eigen::ComputeThinU | Eigen::ComputeThinV);

      Eigen::Matrix<double> uBuff = svds.matrixU().block(0,0,2*M_state,M_state);
      Eigen::Matrix<double> vBuff = svds.matrixV().block(0,0,2*M_state,M_state).transpose();

      aMps.matrix_u(i+M_block,i        ) = uBuff.block(      0,      0,M_state,M_state);
      aMps.matrix_d(i        ,i        ) = uBuff.block(M_state,      0,M_state,M_state);

      bMps.matrix_u(i        ,i        ) = vBuff.block(      0,      0,M_state,M_state);
      bMps.matrix_d(i        ,i+M_block) = vBuff.block(      0,M_state,M_state,M_state);
    }
    // Case 2
    for(size_t i = p1; i < p2; ++i) {
      if(i % nproc != iproc) continue;
      Eigen::Matrix<double> X(2*M_state,2*M_state); X.fill(0.0);
      X.block(      0,      0,M_state,M_state) = uu_ref(i+M_block,i        );
      X.block(      0,M_state,M_state,M_state) = ud_ref(i+M_block,i+M_block);
      X.block(M_state,      0,M_state,M_state) = du_ref(i        ,i        );
      X.block(M_state,M_state,M_state,M_state) = dd_ref(i        ,i+M_block);

      // Canonicalizing 2M -> M+M
      Eigen::JacobiSVD<Matrix<double>> svds(X, Eigen::ComputeThinU | Eigen::ComputeThinV);

      Eigen::Matrix<double> uBuff = svds.matrixU();
      Eigen::Matrix<double> vBuff = svds.matrixV().transpose();

      aMps.matrix_u(i+M_block,i        ) = uBuff.block(      0,      0,M_state,M_state);
      aMps.matrix_u(i+M_block,i+M_block) = uBuff.block(      0,M_state,M_state,M_state);
      aMps.matrix_d(i        ,i        ) = uBuff.block(M_state,      0,M_state,M_state);
      aMps.matrix_d(i        ,i+M_block) = uBuff.block(M_state,M_state,M_state,M_state);

      bMps.matrix_u(i        ,i        ) = vBuff.block(      0,      0,M_state,M_state);
      bMps.matrix_u(i+M_block,i        ) = vBuff.block(M_state,      0,M_state,M_state);
      bMps.matrix_d(i        ,i+M_block) = vBuff.block(      0,M_state,M_state,M_state);
      bMps.matrix_d(i+M_block,i+M_block) = vBuff.block(M_state,M_state,M_state,M_state);
    }
    // Case 3
    for(size_t i = p2; i < p3; ++i) {
      if(i % nproc != iproc) continue;
      Eigen::Matrix<double> X(2*M_state,2*M_state); X.fill(0.0);
      X.block(      0,      0,M_state,M_state) = uu_ref(i+M_block,i        );
      X.block(      0,M_state,M_state,M_state) = ud_ref(i+M_block,i+M_block);
      X.block(M_state,      0,M_state,M_state) = du_ref(i        ,i        );
      X.block(M_state,M_state,M_state,M_state) = dd_ref(i        ,i+M_block);

      // Truncating 2M -> M
      Eigen::JacobiSVD<Matrix<double>> svds(X, Eigen::ComputeThinU | Eigen::ComputeThinV);

      Eigen::Matrix<double> uBuff = svds.matrixU().block(0,0,2*M_state,M_state);
      Eigen::Matrix<double> vBuff = svds.matrixV().block(0,0,2*M_state,M_state).transpose();

      aMps.matrix_u(i+M_block,i+M_block) = uBuff.block(      0,      0,M_state,M_state);
      aMps.matrix_d(i        ,i+M_block) = uBuff.block(M_state,      0,M_state,M_state);

      bMps.matrix_u(i+M_block,i        ) = vBuff.block(      0,      0,M_state,M_state);
      bMps.matrix_d(i+M_block,i+M_block) = vBuff.block(      0,M_state,M_state,M_state);
    }
  }
}
