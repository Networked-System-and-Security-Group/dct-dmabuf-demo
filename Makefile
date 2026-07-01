CC ?= gcc
CUDA_HOME ?= /usr/local/cuda

CFLAGS ?= -O2
IB_LIBS := -libverbs -lmlx5
CUDA_CFLAGS := -I$(CUDA_HOME)/include
CUDA_LIBS := -L$(CUDA_HOME)/lib64 -lcuda -lcudart -Wl,-rpath,$(CUDA_HOME)/lib64

.PHONY: all clean

all: dct_test dct_test_remote dct_test_gpu

dct_test: dct_test.c
	$(CC) $(CFLAGS) -o $@ $< $(IB_LIBS)

dct_test_remote: dct_test_remote.c
	$(CC) $(CFLAGS) -o $@ $< $(IB_LIBS)

dct_test_gpu: dct_test_gpu.c
	$(CC) $(CFLAGS) -o $@ $< $(IB_LIBS) $(CUDA_CFLAGS) $(CUDA_LIBS)

clean:
	rm -f dct_test dct_test_remote dct_test_gpu
