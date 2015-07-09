CC=nvcc
SRCDIR=.
gpuNUFFT_DIR= $(SRCDIR)/gpuNUFFT-2.0.6rc2/CUDA
INCLUDEDIR = $(SRCDIR)/includes $(gpuNUFFT_DIR)/inc
CFLAGS= $(addprefix -I, $(INCLUDEDIR))
LDFLAGS= -lcublas -lgpuNUFFT_f -lgpuNUFFT_ATM_f -L$(gpuNUFFT_DIR)/bin

BINARY=grasp
SRCFILES=utils.c matrix.c grasp.cu
OBJFILES=$(SRCFILES:.c=.o)
gpuNUFFT_FILES=libgpuNUFFT_f.so libgpuNUFFT_ATM_f.so

DOWNLOAD=$(shell which wget)
EXTRACT=$(shell which unzip) -ou
CP=$(shell which cp) -f
RM=$(shell which rm) -vf
MV=$(shell which mv) -f

all: CFLAGS += -O3 -Xlinker -s -Xcompiler -march=native
all: depend convert_dump $(BINARY)

debug: CFLAGS += -DDEBUG -g -G -lineinfo
debug: depend convert_dump $(BINARY)

$(BINARY): gpuNUFFT $(OBJFILES)
	$(CC) $(CFLAGS) $(OBJFILES) -o $@ $(LDFLAGS)

convert_dump: CC=gcc
convert_dump: CFLAGS = -I $(SRCDIR)/includes -I/usr/local/cuda/cuda/include
convert_dump: LDFLAGS =
convert_dump: convert_dump.o utils.o
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o convert_dump

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

.PHONY: clean distclean depend gpuNUFFT extract download cscope

cscope:
	find $(SRCDIR) /usr/local/pkg/cuda/6.5/include -type f \
		-name '*.c' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \
		-o -name '*.cu' > $(SRCDIR)/cscope.in
	cscope -bi $(SRCDIR)/cscope.in
	$(RM) $(SRCDIR)/cscope.in

clean:
	$(RM) *.o

distclean: clean
	$(RM) $(BINARY) convertmat convert_dump
	$(RM) $(addprefix "$(gpuNUFFT_DIR)/bin/", $(gpuNUFFT_FILES))
	find ./gpuNUFFT-2.0.6rc2 -name CMakeCache.txt -delete
	find ./gpuNUFFT-2.0.6rc2 -type d -name CMakeFiles -exec rm -r {} \; -prune

vpath %.o $(SRCDIR)
vpath %.c $(SRCDIR)
vpath %.cu $(SRCDIR)
vpath %.cpp $(SRCDIR)
