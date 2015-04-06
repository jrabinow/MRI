CC=nvcc
CFLAGS=-Iincludes
LDFLAGS=-lcublas
BINARY=grasp

all: CFLAGS += -O3
all: depend $(BINARY)

debug: CFLAGS += -g -DDEBUG
debug: depend $(BINARY)

$(BINARY): grasp.cu
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

.SUFFIXES: .cu

.cu: CC=nvcc
.cu:
	$(CC) -c $^

depend:
	$(CC) $(CFLAGS) -M *.c > .depend

-include .depend

.PHONY: clean distclean depend

clean:
	$(RM) *.o

distclean: clean
	$(RM) $(BINARY)

