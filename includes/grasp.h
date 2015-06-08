/*
 * C/CUDA implementation of GRASP.
 * Emma ????, Felix Moody, & Julien Rabinow
 * Fall 2014-Spring 2015
 */

#ifndef GRASP_H
#define GRASP_H

/* System headers */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>			// square root
#include <time.h>			// benchmarking
#include <complex.h>			// complex double type and operations

/* CUDA headers */
#include <cuda_runtime.h>
#include "cublas_v2.h"			// CUBLAS
#include <cuComplex.h>			// cuDoubleComplex type and operations

/* Project headers */
#include <matrix.h>			// host and device matrix metadata types
#include <cudaErr.h>			// cuda and cublas error handlers
#include <TVTemp.h>			// total variate temporal operator
#include <multicoilGpuNUFFT.hpp>	// multicoil nonuniform FFT operator
#include <utils.h>			/* utility functions */

/* CORRECT LINKING DEMO */
//#include <cuda_utils.hpp>

#define NUM_SPOKES	21
#define NUM_ITERATIONS	8

typedef struct {
	Matrix* read; // k space readings
	int num_spokes; // spokes per frame
	int num_frames; // frames in data
	double lambda; // trade off control TODO: between what?
	double l1Smooth; // TODO: find out what this does
	int num_iter; // number of iterations of the reconstruction
	cublasHandle_t handle; // handle to CUBLAS context
} Param_type;

#define DATA_DIR "./liver_data/"

#define TRAJ_FILE_PATH DATA_DIR "k.matrix"
#define SENS_FILE_PATH DATA_DIR "b1.matrix"
#define READ_FILE_PATH DATA_DIR "kdata.matrix"
#define COMP_FILE_PATH DATA_DIR "w.matrix"


void normalize(Matrix* mat);

void apply_density_compensation(Matrix* read, Matrix* comp);

Matrix* load_matrix_from_file(const char* path, size_t* dims, varFlag vartype);

void load_data(Matrix** traj, Matrix** sens, Matrix** read, Matrix** comp,
		Param_type* param);

Matrix* make_time_series(Matrix* traj, Matrix* read, Matrix* comp, Param_type* param);

#endif /* #ifndef GRASP_H */
