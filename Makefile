CC=nvcc
CFLAGS=-Iincludes #-Wall -Wextra
LDFLAGS=-lcublas
#ifeq ($(CC), gcc)
#	CFLAGS += --short-enums
#endif

all: CFLAGS += -O3
#all: LDFLAGS += -s
all: depend grasp

debug: CFLAGS += -g -DDEBUG
debug: depend grasp

grasp: grasp.cu
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

.SUFFIXES: .cu

.cu: CC=nvcc
.cu:
	$(CC) -c $^

depend:
	$(CC) $(CFLAGS) -M *.c > .depend

-include .depend

compress: $(BINARY)
	gzexe $(BINARY) && $(RM) $(BINARY)~

decompress:
	test -f $(BINARY) && gzexe -d $(BINARY) && $(RM) $(BINARY)~ || $(MAKE)

.PHONY: clean distclean depend

clean:
	$(RM) *.o

distclean: clean
	$(RM) grasp
