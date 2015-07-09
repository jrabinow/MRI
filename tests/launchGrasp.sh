#!/bin/sh
if [ $# -eq 1 ] && [ $1 == '-h' ]; then
	cat << EOF
Usage: $0 [OPTION] options
Options: -g: cuda-gdb options ./grasp
	 -h: print this help message
	 -m: cuda-memcheck options ./grasp
	 -t: time options ./grasp
	 -v: valgrind options ./grasp
EOF
elif [ $# -ge 1 ] && [ $1 == '-g' ]; then
	shift
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/pkg/cuda/6.5/lib64:./gpuNUFFT-2.0.6rc2/CUDA/bin cuda-gdb "$@" ./grasp
elif [ $# -ge 1 ] && [ $1 == '-v' ]; then
	shift
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/pkg/cuda/6.5/lib64:./gpuNUFFT-2.0.6rc2/CUDA/bin valgrind "$@" ./grasp
elif [ $# -ge 1 ] && [ $1 == '-m' ]; then
	shift
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/pkg/cuda/6.5/lib64:./gpuNUFFT-2.0.6rc2/CUDA/bin cuda-memcheck "$@" ./grasp
elif [ $# -ge 1 ] && [ $1 == '-t' ]; then
	shift
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/pkg/cuda/6.5/lib64:./gpuNUFFT-2.0.6rc2/CUDA/bin time "$@" ./grasp
else
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/pkg/cuda/6.5/lib64:./gpuNUFFT-2.0.6rc2/CUDA/bin ./grasp
fi
