#include <iostream>
#include <iomanip>
#include <vector>
#include "tiledarray.h"

int main(int argc, char** argv) {

  namespace TA = TiledArray;

  madness::World& world = madness::initialize(argc,argv);

  std::vector<size_t> tile_boundaries = {0,4,8,12,16,20,24,28,32};

  std::vector<TA::TiledRange1>
    ranges(2, TA::TiledRange1(tile_boundaries.begin(), tile_boundaries.end()));

  TA::TiledRange trange(ranges.begin(), ranges.end());

  std::vector<size_t> shape;
  for(size_t i = 0; i < 8; ++i) shape.push_back(i*8+i);

  TA::Array<double, 2> a(world, trange, shape);

  size_t nproc = world.size();
  size_t iproc = world.rank();

//for(size_t i = 0; i < 8; ++i) {
//  std::vector<size_t> idx = {i,i};
//  if(i%nproc == iproc)
//  a.set(idx,1.0);
//}

  a.set_all_local(1.0);

  world.gop.fence();

  for(auto it = a.begin(); it != a.end(); ++it) {
//  auto ref = it->get();
//  if(!ref.empty()) {
      std::cout << "tile(" << it.index()[0] << "," << it.index()[1] << ") is stored in proc. " << iproc << std::endl;
//    std::cout << ref << std::endl << std::endl;
//  }
  }

//    if(a.is_local(idx))
//      std::cout << "tile(" << i << "," << j << ") is stored in proc. " << iproc << std::endl;

//auto tmp = a.range();
//std::cout << tmp.dim() << "," << tmp.rank() << std::endl;
//auto tmp = a.trange().data()[0].tiles();
//std::cout << tmp.first << "," << tmp.second << std::endl;

  madness::finalize();

  return 0;
}
