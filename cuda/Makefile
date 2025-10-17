NVCC = nvcc
ARCH ?= -arch=sm_75
NVCCFLAGS ?= -O3 $(ARCH) -Wno-deprecated-gpu-targets

all: flash_atten flash_atten_causal

flash_atten: flash_atten.cu
	$(NVCC) $(NVCCFLAGS) -o flash_atten flash_atten.cu

flash_atten_causal: flash_atten_causal.cu
	$(NVCC) $(NVCCFLAGS) -o flash_atten_causal flash_atten_causal.cu

clean:
	rm -f flash_atten flash_atten_causal
