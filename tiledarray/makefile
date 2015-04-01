CXX=mpic++
CXXFLAGS=-O2 -std=c++11

BLASDIR=/home100/opt/intel/mkl
BLASINC=-I$(BLASDIR)/include
BLASLIB=-L$(BLASDIR)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_sequential

TADIR=/homec/naokin/tiledarray
TAINC=-I$(TADIR)/include -I$(TADIR)/include/eigen3
TALIB=$(TADIR)/lib/libMAD*.a

INCLUDEFLAGS= $(BLASINC) $(TAINC)
LIBRARYFLAGS= $(BLASLIB) $(TALIB)

SRC_iTEBD = F_gauge_fix.cpp  GetWavefunction.cpp  imagEvolve.cpp  iTEBD.cpp  MPS_init.cpp  TA_sparse_svd.cpp

OBJ_iTEBD = $(SRC_iTEBD:.cpp=.o)

.cpp.o	:
	$(CXX) $(CXXFLAGS) $(INCLUDEFLAGS) -c $< -o $@

all	: iTEBD.x

iTEBD.x	: $(OBJ_iTEBD)
	$(CXX) $(CXXFLAGS) -o iTEBD.x $(OBJ_iTEBD) $(LIBRARYFLAGS)

clean	:
	rm *.o; rm *.x; rm *.a;
