NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++14 
SRC_DIR = code

# Resolve paths relative to this Makefile, not the shell working directory.
ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SRC_PATH := $(ROOT_DIR)$(SRC_DIR)

# Define the output directory
OUT_DIR = output
OUT_PATH := $(ROOT_DIR)$(OUT_DIR)

# Target executable name
TARGET = $(OUT_PATH)/radix_sort_eval

# Object files
OBJS = $(OUT_PATH)/main.o \
	$(OUT_PATH)/scan.o \
	$(OUT_PATH)/radix_cpu_seq.o \
	$(OUT_PATH)/radix_gpu_seq.o \
	$(OUT_PATH)/radix_gpu_par.o \
	$(OUT_PATH)/radix_gpu_lib.o \
	$(OUT_PATH)/radix_gpu_opt_p2.o \
	$(OUT_PATH)/radix_gpu_opt_p3.o

all: build_dir $(TARGET)

# Create the output directory if it doesn't exist
build_dir:
	mkdir -p $(OUT_PATH)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(OBJS)

$(OUT_PATH)/main.o: $(SRC_PATH)/main.cu $(SRC_PATH)/utils.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC_PATH)/main.cu -o $@

$(OUT_PATH)/radix_cpu_seq.o: $(SRC_PATH)/radix_cpu_seq.cpp $(SRC_PATH)/utils.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC_PATH)/radix_cpu_seq.cpp -o $@

$(OUT_PATH)/radix_gpu_seq.o: $(SRC_PATH)/radix_gpu_seq.cu $(SRC_PATH)/utils.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC_PATH)/radix_gpu_seq.cu -o $@

$(OUT_PATH)/radix_gpu_par.o: $(SRC_PATH)/radix_gpu_par.cu $(SRC_PATH)/utils.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC_PATH)/radix_gpu_par.cu -o $@

$(OUT_PATH)/radix_gpu_lib.o: $(SRC_PATH)/radix_gpu_lib.cu $(SRC_PATH)/utils.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC_PATH)/radix_gpu_lib.cu -o $@

$(OUT_PATH)/radix_gpu_opt_p2.o: $(SRC_PATH)/radix_gpu_opt_p2.cu $(SRC_PATH)/utils.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC_PATH)/radix_gpu_opt_p2.cu -o $@

$(OUT_PATH)/radix_gpu_opt_p3.o: $(SRC_PATH)/radix_gpu_opt_p3.cu $(SRC_PATH)/utils.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC_PATH)/radix_gpu_opt_p3.cu -o $@

$(OUT_PATH)/scan.o: $(SRC_PATH)/scan.cu $(SRC_PATH)/utils.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC_PATH)/scan.cu -o $@

clean:
	rm -rf $(OUT_PATH)