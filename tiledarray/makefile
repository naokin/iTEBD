CXX=mpic++
CXXFLAGS=-g -O3 -std=c++11 -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE

BLASDIR=/home100/opt/intel/mkl
BLASINC=-I$(BLASDIR)/include
BLASLIB=-L$(BLASDIR)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential

TADIR=/homec/naokin/tiledarray
TAINC=-I$(TADIR)/include -I$(TADIR)/include/eigen3
TALIB=$(TADIR)/lib/libMAD*.a

INCLUDEFLAGS= $(BLASINC) $(TAINC)
LIBRARYFLAGS= $(BLASLIB) $(TALIB)

SRC_iTEBD = F_gauge_fix.cpp imagEvolve.cpp realEvolve.cpp MPS_init.cpp SqNorm.cpp TA_sparse_svd.cpp

OBJ_iTEBD = $(SRC_iTEBD:.cpp=.o)

.cpp.o	:
	$(CXX) $(CXXFLAGS) $(INCLUDEFLAGS) -c $< -o $@

all	: iTEBD_imag.x iTEBD_real.x

iTEBD_imag.x	: iTEBD_imag.o $(OBJ_iTEBD)
	$(CXX) $(CXXFLAGS) -o iTEBD_imag.x iTEBD_imag.o $(OBJ_iTEBD) $(LIBRARYFLAGS)

iTEBD_real.x	: iTEBD_real.o $(OBJ_iTEBD)
	$(CXX) $(CXXFLAGS) -o iTEBD_real.x iTEBD_real.o $(OBJ_iTEBD) $(LIBRARYFLAGS)

clean	:
	rm *.o; rm *.x; rm *.a;
