NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++14 

# Define the output directory
OUT_DIR = output

# Target executable name (now inside the output folder)
TARGET = $(OUT_DIR)/radix_sort_eval

# Object files (now explicitly mapped to the output folder)
OBJS = $(OUT_DIR)/main.o \
       $(OUT_DIR)/radix_cpu_seq.o \
       $(OUT_DIR)/radix_gpu_seq.o \
       $(OUT_DIR)/radix_gpu_par.o \
       $(OUT_DIR)/radix_gpu_lib.o \
       $(OUT_DIR)/radix_gpu_opt.o

all: build_dir $(TARGET)

# Create the output directory if it doesn't exist
build_dir:
	mkdir -p $(OUT_DIR)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(OBJS)

$(OUT_DIR)/main.o: main.cu utils.h
	$(NVCC) $(NVCC_FLAGS) -c main.cu -o $@

$(OUT_DIR)/radix_cpu_seq.o: radix_cpu_seq.cpp utils.h
	$(NVCC) $(NVCC_FLAGS) -c radix_cpu_seq.cpp -o $@

$(OUT_DIR)/radix_gpu_seq.o: radix_gpu_seq.cu utils.h
	$(NVCC) $(NVCC_FLAGS) -c radix_gpu_seq.cu -o $@

$(OUT_DIR)/radix_gpu_par.o: radix_gpu_par.cu utils.h
	$(NVCC) $(NVCC_FLAGS) -c radix_gpu_par.cu -o $@

$(OUT_DIR)/radix_gpu_lib.o: radix_gpu_lib.cu utils.h
	$(NVCC) $(NVCC_FLAGS) -c radix_gpu_lib.cu -o $@

$(OUT_DIR)/radix_gpu_opt.o: radix_gpu_opt.cu utils.h
	$(NVCC) $(NVCC_FLAGS) -c radix_gpu_opt.cu -o $@

# Clean now just deletes the entire output directory
clean:
	rm -rf $(OUT_DIR)