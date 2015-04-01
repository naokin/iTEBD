#!/bin/bash

SRC=$1
EXE=${SRC%.*}.x

mpic++  -std=c++11 \
        -I/homec/naokin/tiledarray/include \
        -I/homec/naokin/tiledarray/include/eigen3 \
        ${SRC} -o ${EXE} \
        /homec/naokin/tiledarray/lib/libMAD*.a \
        -L/home100/opt/intel/mkl/lib/intel64 \
        -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

#
