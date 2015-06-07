CC=nvcc
gpuNUFFT_DIR=./gpuNUFFT-2.0.6rc2/CUDA
CFLAGS= -I$(gpuNUFFT_DIR)/inc -L$(gpuNUFFT_DIR)/bin
LDFLAGS=-lcublas -lgpuNUFFT_f -lgpuNUFFT_ATM_f

BINARY=grasp
SRCFILES=utils.c grasp.cu
OBJFILES=$(SRCFILES:.c=.o)
gpuNUFFT_FILES=libgpuNUFFT_f.so libgpuNUFFT_ATM_f.so

DOWNLOAD=$(shell which wget)
EXTRACT=$(shell which unzip) -ou
CP=$(shell which cp) -f
RM=$(shell which rm) -vf


all: CFLAGS += -O3
all: depend $(BINARY)

debug: CFLAGS += -g -G -DDEBUG
debug: depend $(BINARY)

$(BINARY): gpuNUFFT $(OBJFILES)
	$(CC) $(CFLAGS) $(OBJFILES) -o $@ $(LDFLAGS)

%.o: CC=g++

gpuNUFFT: extract
	(test -z '' $(addprefix -a -f $(gpuNUFFT_DIR)/bin/, $(gpuNUFFT_FILES))) ||\
	(mkdir -p "$(gpuNUFFT_DIR)/build" && \
	cd "$(gpuNUFFT_DIR)/build" && \
	cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DMATLAB_ROOT_DIR=/opt/matlab \
	-UCMAKE_CXX_COMPILER -UCMAKE_C_COMPILER && \
	$(MAKE) -j && \
	cd -)

extract: download
	test -d gpuNUFFT-2.0.6rc2 || \
		$(EXTRACT) gpuNUFFT-2.0.6.zip && \
		$(RM) gpuNUFFT-2.0.6.zip

download:
	test -d gpuNUFFT-2.0.6rc2 || test -f gpuNUFFT-2.0.6.zip || \
		$(DOWNLOAD) 'http://cai2r.net/sites/default/files/software/gpuNUFFT-2.0.6.zip'

.PHONY: clean distclean depend gpuNUFFT copy_files

clean:
	$(RM) *.o

distclean: clean
	$(RM) $(BINARY)
	$(RM) $(addprefix "$(gpuNUFFT_DIR)/bin/", $(gpuNUFFT_FILES))
	find ./gpuNUFFT-2.0.6rc2 -name CMakeCache.txt -delete 2>/dev/null
	# the following command works correctly but prints error messages to stdout. We squash them
	find ./gpuNUFFT-2.0.6rc2 -type d -name CMakeFiles -exec rm -r {} \; 2>/dev/null

