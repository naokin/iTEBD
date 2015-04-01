#include <iostream>
#include <iomanip>
#include <vector>
#include "tiledarray.h"

int main(int argc, char** argv) {

  madness::World& world = madness::initialize(argc,argv);

  size_t iproc = world.rank();
  size_t nproc = world.size();

  std::vector<size_t> v(nproc,0);
  v[iproc] = iproc;

  std::cout << "[B] process[" << iproc << "] :: ";
  for(size_t i = 0; i < v.size(); ++i) std::cout << v[i] << " "; std::cout << std::endl;

  world.gop.sum(v.data(),nproc);

  std::cout << "[A] process[" << iproc << "] :: ";
  for(size_t i = 0; i < v.size(); ++i) std::cout << v[i] << " "; std::cout << std::endl;

  TiledArray::Tensor<double> x;

  if(iproc == 0) {
    x = TiledArray::Tensor<double>(TiledArray::Range(4,4),1.0);
    std::cout << "printing sending Tensor object :: " << std::endl;
    std::cout << x << std::endl;
    world.gop.send(nproc-1,0,x);
  }

  if(iproc == nproc-1) {
    x = world.gop.recv<TiledArray::Tensor<double>>(0,0);
    std::cout << "printing received Tensor object :: " << std::endl;
    std::cout << x << std::endl;
  }

  madness::finalize();

  return 0;
}
