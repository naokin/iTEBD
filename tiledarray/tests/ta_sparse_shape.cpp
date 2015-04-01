#include <iostream>
#include <tiledarray.h>

int main(int argc, char** argv) {
  // Initialize runtime
  madness::World& world = madness::initialize(argc, argv);

  const size_t M_state = 100;

  std::vector<int> aQ; aQ.reserve(M_spin+1);
  std::vector<int> bQ; bQ.reserve(M_spin+1);

  // define spin symmetry sector

  for(int k = -M_spin  ; k <= M_spin; k+=2) aQ.push_back(k);
  for(int k = -M_spin+1; k <  M_spin; k+=2) bQ.push_back(k);

  // matrix range

  std::vector<size_t> aBlock; aBlock.reserve(aQ.size()+1);
  for(size_t i = 0; i <= M_state*aQ.size(); i += M_state) aBlock.push_back(i);

  std::vector<size_t> bBlock; bBlock.reserve(bQ.size()+1);
  for(size_t i = 0; i <= M_state*bQ.size(); i += M_state) bBlock.push_back(i);

  std::vector<TA::TiledRange1> aMatrixRange(2);
  aMatrixRange[0] = TA::TiledRange1(bBlock.begin(), bBlock.end());
  aMatrixRange[1] = TA::TiledRange1(aBlock.begin(), aBlock.end());

  std::vector<TA::TiledRange1> bMatrixRange(2);
  bMatrixRange[0] = TA::TiledRange1(aBlock.begin(), aBlock.end());
  bMatrixRange[1] = TA::TiledRange1(bBlock.begin(), bBlock.end());

  // shape

  for(size_t i = 0; i < bQ.size(); ++i) {
    for(size_t j = 0; j < aQ.size(); ++j) {
      // spin up component
      if(bQ[i]+1 == aQ[i]) {
        size_t index[2] = {i, j};
        
      }
    }
  }
  








  // Construct TiledRange
  std::vector<unsigned int> matrix_blocking;
  matrix_blocking.reserve(num_blocks + 1);
  for(long i = 0; i <= matrix_size; i += block_size)
    matrix_blocking.push_back(i);

  std::vector<unsigned int> coeff_blocking;
  coeff_blocking.reserve(coeff_num_blocks + 1);
  for(long i = 0; i <= coeff_size; i += coeff_block_size)
    coeff_blocking.push_back(i);

  std::vector<unsigned int> df_blocking;
  df_blocking.reserve(df_num_blocks + 1);
  for(long i = 0; i <= df_size; i += df_block_size)
    df_blocking.push_back(i);

  std::vector<TiledArray::TiledRange1> matrix_blocking2(
    2, TiledArray::TiledRange1(matrix_blocking.begin(), matrix_blocking.end())
  );

  // Create C^T blocking
  std::vector<TiledArray::TiledRange1> coeff_blocking2;
  coeff_blocking2.reserve(2);
  coeff_blocking2.push_back(TiledArray::TiledRange1(matrix_blocking.begin(), matrix_blocking.end()));
  coeff_blocking2.push_back(TiledArray::TiledRange1(coeff_blocking.begin(), coeff_blocking.end()));

  std::vector<TiledArray::TiledRange1> df_blocking2;
  df_blocking2.reserve(3);
  df_blocking2.push_back(TiledArray::TiledRange1(matrix_blocking.begin(), matrix_blocking.end()));
  df_blocking2.push_back(TiledArray::TiledRange1(matrix_blocking.begin(), matrix_blocking.end()));
  df_blocking2.push_back(TiledArray::TiledRange1(df_blocking.begin(), df_blocking.end()));

  std::vector<TiledArray::TiledRange1> temp_blocking2;
  temp_blocking2.reserve(3);
  temp_blocking2.push_back(TiledArray::TiledRange1(coeff_blocking.begin(), coeff_blocking.end()));
  temp_blocking2.push_back(TiledArray::TiledRange1(matrix_blocking.begin(), matrix_blocking.end()));
  temp_blocking2.push_back(TiledArray::TiledRange1(df_blocking.begin(), df_blocking.end()));


  TiledArray::TiledRange matrix_trange(matrix_blocking2.begin(), matrix_blocking2.end());
  TiledArray::TiledRange coeff_trange(coeff_blocking2.begin(), coeff_blocking2.end());
  TiledArray::TiledRange df_trange(df_blocking2.begin(), df_blocking2.end());
  TiledArray::TiledRange temp_trange(temp_blocking2.begin(), temp_blocking2.end());

  // Construct and initialize arrays
  TiledArray::Array<double, 2> C(world, coeff_trange);
  TiledArray::Array<double, 2> G(world, matrix_trange);
  TiledArray::Array<double, 2> H(world, matrix_trange);
  TiledArray::Array<double, 2> D(world, matrix_trange);
  TiledArray::Array<double, 2> F(world, matrix_trange);
  TiledArray::Array<double, 3> Eri(world, df_trange);
  TiledArray::Array<double, 3> K_temp(world, temp_trange);
  C.set_all_local(1.0);
  D.set_all_local(1.0);
  H.set_all_local(1.0);
  F.set_all_local(1.0);
  G.set_all_local(1.0);
  Eri.set_all_local(1.0);
  world.gop.fence();


  // Start clock
  world.gop.fence();
  const double wall_time_start = madness::wall_time();

  // Do fock build
  for(int i = 0; i < repeat; ++i) {

    K_temp("j,Z,P") = C("m,Z") * Eri("m,j,P");

    // Compute coulomb and exchange
    G("i,j") = 2.0 * ( Eri("i,j,P") * ( C("m,Z") * K_temp("m,Z,P") ) )
                   - ( K_temp("i,Z,P") * K_temp("j,Z,P") );
    D("mu,nu") = C("mu,i") * C("nu,i");

    F("i,j") = G("i,j") + H("i,j");

    world.gop.fence();
    if(world.rank() == 0)
      std::cout << "Iteration " << i + 1 << "\n";
  }

  // Stop clock
  const double wall_time_stop = madness::wall_time();

  const double total_time = wall_time_stop - wall_time_start;

  double gflops = 2.0 * double(coeff_size * matrix_size * matrix_size * df_size); // C("Z,m") * Eri("m,n,P") = K_temp("Z,n,P")
  gflops += 2.0 * double(coeff_size * matrix_size * df_size); // C("Z,n") * K_temp("Z,n,P") = temp("P")
  gflops += 2.0 * double(matrix_size * matrix_size * df_size); // Eri("i,j,P") * temp("P") = Final("i,j")
  gflops += 2.0 * double(coeff_size * matrix_size * matrix_size * df_size); // K_temp("Z,i,P") * K_temp("Z,j,P")
  gflops += 1.0 * double(matrix_size * matrix_size); // 2 * J("i,j") - K("i,j")
  gflops += 2.0 * double(coeff_size * matrix_size * matrix_size); // C("Z,mu") * C("Z,nu")
  gflops += double(matrix_size); // G("i,j") + H("i,j")
  gflops = double(repeat * gflops)/(1e9 * total_time);

  if(world.rank() == 0){
    std::cout << "Average wall time = " << (wall_time_stop - wall_time_start) / double(repeat) << std::endl;
    std::cout << "Memory needed (not including undeclared temporaries) = " <<
            4 * matrix_memory + coeff_memory + tensor_memory + co_tensor_memory << " GB" << std::endl;
    std::cout << "GFlops = " << gflops << std::endl;
  }

  madness::finalize();
  return 0;
}
