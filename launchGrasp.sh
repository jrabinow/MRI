#!/bin/sh
if [ $# -ge 1 ] && [ $1 == '-d' ]; then
	shift
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/pkg/cuda/6.5/lib64:./gpuNUFFT-2.0.6rc2/CUDA/bin cuda-gdb "$@" ./grasp
else
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/pkg/cuda/6.5/lib64:./gpuNUFFT-2.0.6rc2/CUDA/bin ./grasp
fi
