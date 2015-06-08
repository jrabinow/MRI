/* Multicoil Wrapper for gpuNUFFT library */

#ifndef MULTICOIL_GPUNUFFT_H
#define MULTICOIL_GPUNUFFT_H

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <matrix.h>
#include <cudaErr.h>
#include <utils.h>

extern void createMulticoilGpuNUFFTOperator(Matrix * traj,
	Matrix * comp, Matrix * sens, int kernel_width, int sector_width,
	double oversample_ratio);

#endif
