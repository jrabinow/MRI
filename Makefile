CC=nvcc
gpuNUFFT_DIR=./gpuNUFFT-2.0.6rc2/CUDA
CFLAGS=-Iincludes -I$(gpuNUFFT_DIR)/inc
LDFLAGS=-lcublas


BINARY=grasp
SRCFILES=grasp.cu
gpuNUFFT_FILES=libgpuNUFFT_f.so libgpuNUFFT_ATM_f.so

DOWNLOAD=$(shell which wget)
EXTRACT=$(shell which unzip) -ou
CP=$(shell which cp) -f


all: CFLAGS += -O3
all: depend $(BINARY)

debug: CFLAGS += -g -DDEBUG
debug: depend $(BINARY)

# removed gpuNUFFT prerequisite to prevent building at each compilation
#$(BINARY): gpuNUFFT $(SRCFILES)
$(BINARY): $(SRCFILES)
	$(CC) $(CFLAGS) $(LDFLAGS) $(addprefix "$(gpuNUFFT_DIR)/bin/", $(gpuNUFFT_FILES)) $(SRCFILES) -o $@

#copy_files:
#	$(foreach shared_lib, $(gpuNUFFT_FILES), test -f $(shared_lib) || $(CP) $(addprefix "$(gpuNUFFT_DIR)/bin/", $(shared_lib)) .)

gpuNUFFT: extract
	cd "$(gpuNUFFT_DIR)/build" && cmake .. -DMATLAB_ROOT_DIR=/opt/matlab && $(MAKE) -j

extract: download
	test -d gpuNUFFT-2.0.6rc2 || $(EXTRACT) gpuNUFFT-2.0.6.zip && $(RM) gpuNUFFT-2.0.6.zip

download:
	test -f gpuNUFFT-2.0.6.zip || $(DOWNLOAD) 'http://cai2r.net/sites/default/files/software/gpuNUFFT-2.0.6.zip'

.SUFFIXES: .cu

.cu: CC=nvcc
#.cu:
#	$(CC) -c $^

#depend:
#	$(CC) $(CFLAGS) -M *.c > .depend

#-include .depend

.PHONY: clean distclean depend unzip download build_gpuNUFFT gpuNUFFT

clean:
	$(RM) *.o $(gpuNUFFT_FILES)

distclean: clean
	$(RM) $(BINARY)

