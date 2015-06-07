/*
 * Allocation, Cuda and CuBLAS error handers and macro wrappers
 */

#ifndef cudaErr
#define cudaErr

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

// error handlers wrappers
#define allocErrChk(err) { allocErrorHandler(err, __FILE__, __LINE__); }
#define cudaErrChk(err) { cudaErrorHandler(err, __FILE__, __LINE__); }
#define cublasErrChk(err) { cublasErrorHandler(err, __FILE__, __LINE__); }

// error handlers
inline void * allocErrorHandler(void * return_ptr, const char * file, int line) {
	if (return_ptr == NULL) {
		fprintf(stderr,
				"Allocation error in %s on line %d\n",
				file,
				line);
		exit(EXIT_FAILURE);
	}
	return return_ptr;
}

inline void cudaErrorHandler(cudaError_t err, const char * file, int line) {
	if (err != cudaSuccess) {
		fprintf(stderr,
				"CUDA Error in %s on line %d:%s\n",
				file,
				line,
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

inline void cublasErrorHandler(cublasStatus_t err, const char * file, int line) {
	if (err != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr,
				"CUBLAS Error in %s on line $d:%s\n",
				file,
				line,
				(const char *)err);
		exit(EXIT_FAILURE);
	}
}

#endif
