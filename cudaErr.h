/*
 * Cuda and CuBLAS error handers and macro wrappers
 */

#ifndef cudaErr
#define cudaErr

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

// cuda and cublas error handlers wrappers
#define cudaErrChk(err) { cudaErrorHandler(err, __FILE__, __LINE__); }
#define cublasErrChk(err) {cublasErrorHandler(err, __FILE__, __LINE__); }

// cuda and cublas error handlers
inline void cudaErrorHandler(cudaError_t err, const char * file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s on line %d:%s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
inline void cublasErrorHandler(cublasStatus_t err, const char * file, int line) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS Error in %s on line $d:%s\n", file, line, (const char *)err);
        exit(EXIT_FAILURE);
    }
}

#endif