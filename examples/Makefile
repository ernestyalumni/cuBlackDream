# Ernest Yeung
# ernestyalumni@gmail.com
# 20171015
NVCC = nvcc
NVCC_FLAGS = -std=c++14

GPP = g++
GPP_FLAGS = -std=c++14

LIB_FLAGS = -lcublas 

# FileIOobjects also includes non-CUDA, just C++ code
FileIOobjects    = FileIO.o

# Axonobjects also includes CUDA C++ classes
Axonobjects     = Axon.o

#all: main.exe

linreg.exe: linreg.o $(FileIOobjects) $(Axonobjects) 
	$(NVCC) $(NVCC_FLAGS) $(LIB_FLAGS) $^ -o $@ 
		
linreg.o: linreg.cu 
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

	

# FileIO
FileIO.o: ../src/FileIO/FileIO.cpp
	$(GPP) $(GPP_FLAGS) -c $< -o $@

# Axon 
Axon.o: ../src/Axon/Axon.cu
	$(NVCC) $(NVCC_FLAGS) $(LIB_FLAGS) -dc $< -o $@  

		
clean:
		rm -f *.o *.exe
