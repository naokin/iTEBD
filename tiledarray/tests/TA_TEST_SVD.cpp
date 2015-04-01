#include <iostream>
#include <iomanip>
#include <vector>
#include "tiledarray.h"
#include "TA_sparse_svd.h"

int main(int argc, char** argv) {

  namespace TA = TiledArray;

  madness::World& world = madness::initialize(argc,argv);

  std::vector<size_t> tile_boundaries = {0,4,8,12,16,20,24,28,32};

  std::vector<TA::TiledRange1>
    ranges(2, TA::TiledRange1(tile_boundaries.begin(), tile_boundaries.end()));

  TA::TiledRange trange(ranges.begin(), ranges.end());

  TA::Array<double, 2> a(world, trange);

  size_t myRank = world.rank();

  for(size_t i = 0; i < 8; ++i) {
    for(size_t j = 0; j < 8; ++j) {
      std::vector<size_t> idx = {i,j};
      if(a.is_local(idx))
        std::cout << "tile(" << i << "," << j << ") is stored in proc. " << myRank << std::endl;
    }
  }

  madness::finalize();

  return 0;
}
