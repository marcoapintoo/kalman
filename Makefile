# https://wiki.gentoo.org/wiki/GCC_optimization/es
CXX             := g++
#CXX_FLAGS := -Wall -Wextra -std=c++17 -ggdb
#CXX_FLAGS       := -w -std=c++17 -ggdb
CXX_FLAGS       := -w -std=c++17
CYTHON          := cython
CYTHON_FLAGS    :=  -3 --embed --cplus

BIN             := bin
SRC             := src/cpp
PYSRC           := src/python
PYSRCOUT        := build/cython
INCLUDE         := include
LIB             := lib

LIBRARIES        := -march=native -O3 -ftree-vectorize 
LIBRARIES        += -DARMA_DONT_USE_WRAPPER -DARMA_USE_BLAS -DARMA_USE_LAPACK -DARMA_USE_HDF5
LIBRARIES        += -DARMA_NO_DEBUG
LIBRARIES        += -DNDEBUG
LIBRARIES        += -lopenblas -llapack -lhdf5 -lfftw3
LIBRARIES        += -I"d:\ProgramData\Anaconda3\include" -I"d:\ProgramData\Anaconda3\Lib\site-packages\numpy\core\include" -L"d:\ProgramData\Anaconda3\Lib"  -L"d:\ProgramData\Anaconda3\libs" -lpython36 

EXECUTABLE       := state_space_model
PYMODULE         := state_space_model.pyd

TEST_PYTHON_MODULE := $(PYSRC)/test_state_space_model.py
TEST_PYTHON_MODULE := tests/python/module/example.py

CYTHON_PYX_FILES := $(wildcard $(PYSRC)/*.pyx)
CYTHON_CPP_FILES := $(patsubst $(PYSRC)/%.pyx,$(PYSRCOUT)/%.cpp,$(CYTHON_PYX_FILES))
#CYTHON_BIN_FILES := $(patsubst $(PYSRC)/%.c,$(PYSRCOUT)/%,$(CYTHON_PYX_FILES))

#conda install -c anaconda libpython 
###export PYTHONHOME="D:\ProgramData\Anaconda3\"
###export PYTHONPATH="/D/ProgramData/Anaconda3/:/D/ProgramData/Anaconda3/DLLs:/D/ProgramData/Anaconda3/Lib:/D/ProgramData/Anaconda3/Lib/site-packages"

.PHONY: all

all: clean $(BIN)/$(PYMODULE)# $(BIN)/$(EXECUTABLE)
#all: clean $(BIN)/$(EXECUTABLE)

pytest:
	python -u $(TEST_PYTHON_MODULE)

maintest:
	./$(BIN)/$(EXECUTABLE)

#run: clean all pytest maintest
run: clean all maintest

$(BIN)/$(EXECUTABLE): $(SRC)/*.cpp
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE) -L$(LIB) $^ -o $@ $(LIBRARIES)

$(BIN)/$(PYMODULE): $(CYTHON_CPP_FILES)
	$(CXX) $(CXX_FLAGS) -I$(INCLUDE) -L$(LIB) -fPIC $^ -shared -o $@ $(LIBRARIES) 

$(PYSRCOUT)/%.cpp:
	-mkdir $(PYSRCOUT)
	$(CYTHON) $(CYTHON_FLAGS) -o $(PYSRCOUT)/$*.cpp $(PYSRC)/$*.pyx

clean:
	-rm $(PYSRCOUT)/*.cpp
	-rm $(BIN)/*
