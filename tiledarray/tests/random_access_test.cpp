#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include "tiledarray.h"

int main(int argc, char** argv) {

  namespace TA = TiledArray;

  madness::World& world = madness::initialize(argc,argv);

  std::vector<size_t> tile_boundaries = {0,4,8,12,16,20,24,28,32};
  std::vector<TA::TiledRange1> ranges(2, TA::TiledRange1(tile_boundaries.begin(), tile_boundaries.end()));
  TA::TiledRange trange(ranges.begin(), ranges.end());

  TA::Tensor<float> shape(TA::Range(8,8),0.0);
  for(size_t i = 0; i < 8; ++i) shape[i*8+i] = 1.0;

  TA::Array<double, 2, TA::Tensor<double>, TA::SparsePolicy> a(world,trange,TA::SparseShape<float>(shape,trange));
//TA::Array<double, 2, TA::Tensor<double>, TA::SparsePolicy> b;
//TA::Array<double, 2, TA::Tensor<double>, TA::SparsePolicy> c;

  size_t nproc = world.size();
  size_t iproc = world.rank();

//a.set_all_local(1.0);

//for(size_t i = 0; i < 8; ++i) {
//  std::vector<size_t> idx = {i,i};
//  TA::Tensor<double> tmp(TA::Range(4,4),1.0);
//  if(a.is_local(idx)) a.set(idx,tmp);
//}

//b = a;
//c("i,j") = a("i,k")*b("k,j");

  world.gop.fence();

  if(1) {

  std::vector<std::pair<size_t,TA::Tensor<double>>> tasks;

//for(auto it = a.begin(); it != a.end(); ++it) {
//  tasks.push_back(std::make_pair(it.ordinal(),*it));
//  tasks.push_back(std::make_pair(it->get().range().start()[0],*it));
//  tasks.push_back(std::make_pair(it.make_range().finish()[0],*it));
//  std::fill(it->get().begin(),it->get().end(),2.0);
//}

  for(size_t i = 0; i < 8; ++i) {
    for(size_t j = 0; j < 8; ++j) {
      std::vector<size_t> idx = {i,j};
      if(!a.is_zero(idx)) {
        std::cout << "proc = " << iproc << " :: " << i << "," << j << " is seen to be non-zero" << std::endl;
      }
    }
  }

  world.gop.fence();

  for(size_t i = 0; i < 8; ++i) {
    for(size_t j = 0; j < 8; ++j) {
      std::vector<size_t> idx = {i,j};
      if(a.is_local(idx)) {
        std::cout << "proc = " << iproc << " :: " << i << "," << j << " is local" << std::endl;
      }
    }
  }

  world.gop.fence();

  for(size_t i = 0; i < 8; ++i) {
    for(size_t j = 0; j < 8; ++j) {
      std::vector<size_t> idx = {i,j};
      size_t iown = a.owner(idx);
      if(iown == iproc)
        std::cout << "proc = " << iproc << " :: " << i << "," << j << " is owned " << std::endl;
    }
  }

  world.gop.fence();

//if(world.rank() == 0) {
//  std::cout << a.trange().tiles().size()[0] << std::endl;
//  std::cout << a.trange().elements().size()[0] << std::endl;
//}

//for(size_t i = 0; i < tasks.size(); ++i) {
//  std::cout << "tile[" << tasks[i].first << "] is stored on " << iproc << " :: " << tasks[i].second << std::endl;
//}

//auto tmp = a.range();
//std::cout << tmp.dim() << "," << tmp.rank() << std::endl;
//auto tmp = a.trange().data()[0].tiles();
//std::cout << tmp.first << "," << tmp.second << std::endl;

  }

  madness::finalize();

  return 0;
}
