NVCC=nvcc
NVCC_FLAGS=-O2 -arch=sm_61
NVCC_INC_FLAGS=-I/chalmers/sw/sup64/cuda_toolkit-11.2.2/include
NVCC_LIB_DIR=-L/chalmers/sw/sup64/cuda_toolkit-11.2.2/lib64
NVCC_LIBS=-lcudart

all: cuda_mm cuda_mm_tile

cuda_mm.o: cuda_mm.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC_FLAGS) -c cuda_mm.cu -o cuda_mm.o
cuda_mm_tile.o: cuda_mm_tile.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC_FLAGS) -DLINEAR_REDUCTION -c cuda_mm_tile.cu -o cuda_mm_tile.o


cuda_mm: cuda_mm.o
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC_FLAGS) -o cuda_mm $< $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)
cuda_mm_tile: cuda_mm_tile.o
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC_FLAGS) -o cuda_mm_tile $< $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)


build_pt1: cuda_mm
run_pt1:
	./cuda_mm

build_pt2: cuda_mm_tile

	ncu ./cuda_mm_tile

clean:
	rm *.o cuda_mm cuda_mm_tile