#!/bin/sh

/homec/naokin/gnu/gcc/4.8.2/bin/g++ -std=c++11 -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE -I /home100/opt/intel/mkl/include *.cpp -o iTEBD.x -L /home100/opt/intel/lib/intel64 -L /home100/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread

