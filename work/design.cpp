

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
  // M_state = dim.(XX)
  //----+----+--------------+--------------+--------------+--------------+
  //    |    |      +3      |      +1      |      -1      |      -3      |
  //----+----+----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    | XX |    |    |    |    |    |    |    |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    | +2 |    | XX |    |    |    |    |    |    |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    | XX |    |    |    |    |    |    |    |    |    |
  //    +----+----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    | XX |    |    |    |    |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    | +0 |    |    |    |    | XX |    |    |    |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    |    |    | XX |    |    |    |    |    |    |
  // +1 +----+----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    | XX |    |    |    |    |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    | -0 |    |    |    |    | XX |    |    |    |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    |    |    | XX |    |    |    |    |    |    |
  //    +----+----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    |    |    |    | XX |    |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    | -2 |    |    |    |    |    |    |    | XX |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    |    |    |    |    |    | XX |    |    |    |
  //----+----+----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    | XX |    |    |    |    |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    | +2 |    |    |    |    | XX |    |    |    |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    |    |    | XX |    |    |    |    |    |    |
  //    +----+----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    |    |    |    | XX |    |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    | +0 |    |    |    |    |    |    |    | XX |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    |    |    |    |    |    | XX |    |    |    |
  // -1 +----+----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    |    |    |    | XX |    |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    | -0 |    |    |    |    |    |    |    | XX |    |    |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    |    |    |    |    |    | XX |    |    |    |
  //    +----+----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    |    |    |    |    |    |    | XX |    |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    | -2 |    |    |    |    |    |    |    |    |    |    | XX |    |
  //    |    +----+----+----+----+----+----+----+----+----+----+----+----+
  //    |    |    |    |    |    |    |    |    |    |    |    |    | XX |
  //----+----+----+----+----+----+----+----+----+----+----+----+----+----+

  assert(M_sector%2 == 0);

  size_t nRange = M_sector*M_block;

  std::vector<size_t> rangeMps(nRange+1);
  for(size_t i = 0; i <= nRange; ++i) {
    rangeMps[i] = i*M_state;
  }

  size_t nRangeHf = nRange/2;

  MPS<double> aMps;
  aMps.matrix_u.resize(rangeMps,rangeMps);
  aMps.matrix_d.resize(rangeMps,rangeMps);

  // Create spin-up matrix
  for(size_t i = 0; i < nRangeHf; ++i) {
    if(!aMps.matrix_u.shape(i,i)) continue;
    MatrixXd data = MatrixXd::Random(M_state,M_state);
    aMps.matrix_u(i,i).assign(data);
  }
  for(size_t i = 0; i < nRangeHf; ++i) {
    if(!aMps.matrix_u.shape(i+nRangeHf,i+nRangeHf-M_block)) continue;
    MatrixXd data = MatrixXd::Random(M_state,M_state);
    aMps.matrix_u(i+nRangeHf,i+nRangeHf-M_block).assign(data);
  }
  // Create spin-down matrix
  for(size_t i = 0; i < nRangeHf; ++i) {
    if(!aMps.matrix_d.shape(i,i+M_block)) continue;
    MatrixXd data = MatrixXd::Random(M_state,M_state);
    aMps.matrix_d(i,i+M_block).assign(data);
  }
  for(size_t i = 0; i < nRangeHf; ++i) {
    if(!aMps.matrix_d.shape(i+nRangeHf,i+nRangeHf)) continue;
    MatrixXd data = MatrixXd::Random(M_state,M_state);
    aMps.matrix_d(i+nRangeHf,i+nRangeHf).assign(data);
  }

  MPS<double> bMps;
  bMps.matrix_u.resize(rangeMps,rangeMps);
  bMps.matrix_d.resize(rangeMps,rangeMps);

  // Create spin-up matrix
  for(size_t i = 0; i < nRangeHf; ++i) {
    if(!bMps.matrix_u.shape(i,i)) continue;
    MatrixXd data = MatrixXd::Random(M_state,M_state);
    bMps.matrix_u(i,i).assign(data);
  }
  for(size_t i = 0; i < nRangeHf; ++i) {
    if(!bMps.matrix_u.shape(i+nRangeHf,i+nRangeHf-M_block)) continue;
    MatrixXd data = MatrixXd::Random(M_state,M_state);
    bMps.matrix_u(i+nRangeHf,i+nRangeHf-M_block).assign(data);
  }
  // Create spin-down matrix
  for(size_t i = 0; i < nRangeHf; ++i) {
    if(!bMps.matrix_d.shape(i,i+M_block)) continue;
    MatrixXd data = MatrixXd::Random(M_state,M_state);
    bMps.matrix_d(i,i+M_block).assign(data);
  }
  for(size_t i = 0; i < nRangeHf; ++i) {
    if(!bMps.matrix_d.shape(i+nRangeHf,i+nRangeHf)) continue;
    MatrixXd data = MatrixXd::Random(M_state,M_state);
    bMps.matrix_d(i+nRangeHf,i+nRangeHf).assign(data);
  }

  for(size_t t = 0; t < nStep; ++t) {
    imagEvolve(1,aMps,bMps,J,Jz,Hz,dt,M_sector,M_block,M_state);
    imagEvolve(0,bMps,aMps,J,Jz,Hz,dt,M_sector,M_block,M_state);
  }
}

void imagEvolve
(
  bool isForward,
  MPS<double>& aMps,
  MPS<double>& bMps,
  double J, ///< J  coupling constant in 1D-Heisenberg spin hamiltonian
  double Jz,///< Jz coupling constant ...
  double Hz,///< external magnetic field
  double dt,///< step size of time-evolution
  size_t M_sector, ///< Number of symmetry sectors
  size_t M_block, ///< Number of blocks for each symmetry sector
  size_t M_state ///< Number of dense states for each block
)
{
  Wavefunction<double> wfn;

  // Compute aMps*bMps

  wfn.matrix_uu.resize(aMps.matrix_u.range(0),bMps.matrix_u.range(1));
  wfn.matrix_uu += aMps.matrix_u*bMps.matrix_u;

  wfn.matrix_ud.resize(aMps.matrix_u.range(0),bMps.matrix_d.range(1));
  wfn.matrix_ud += aMps.matrix_u*bMps.matrix_d;

  wfn.matrix_du.resize(aMps.matrix_d.range(0),bMps.matrix_u.range(1));
  wfn.matrix_du += aMps.matrix_d*bMps.matrix_u;

  wfn.matrix_dd.resize(aMps.matrix_d.range(0),bMps.matrix_d.range(1));
  wfn.matrix_dd += aMps.matrix_d*bMps.matrix_d;

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
    Wavefunction<double>::matrix_type ud_tmp;
    ud_tmp.resize(aMps.matrix_u.range(0),bMps.matrix_d.range(1));
    ud_tmp += wfn.matrix_ud*coshJ/expJz-wfn.matrix_du*sinhJ/expJz;

    Wavefunction<double>::matrix_type du_tmp;
    du_tmp.resize(aMps.matrix_u.range(0),bMps.matrix_d.range(1));
    du_tmp += wfn.matrix_du*coshJ/expJz-wfn.matrix_ud*sinhJ/expJz;

    wfn.matrix_uu *= ( expJz*expHz);
    wfn.matrix_ud = ud_tmp;
    wfn.matrix_du = du_tmp;
    wfn.matrix_dd *= ( expJz/expHz);
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
