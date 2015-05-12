/*
 * Matrix header test
 * To compile:
 *  $nvcc matrix_test.cu -g -o matrix_test
 *
 * Then to run:
 *  $./matrix_test newHostC
 * or,
 *  $./matrix_test newDeviceC
 * or,
 *  $./matrix_test toDeviceC
 * or,
 *  $./matrix_test crop
 * or,
 *  $./matrix_test cropC
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuComplex.h> // CUDA complex numbers and operations
#include "matrix.h"
#include "cudaErr.h"


void test_new_matrixC_host() {
	// define dimensions of matrix
	// only need to specify the dimensions we need, up to MAX_DIMS
	size_t dims[MAX_DIMS] = {768, 600};

	// allocate a host matrix with those dimensions
	matrixC * mat = new_matrixC(dims, host); 
	
	// initialize matrix data
	for (size_t i = 0; i < mat->num; i++) {
		mat->data[i] = make_cuDoubleComplex(
				(double)i,
				(double)(i*2));
	}

	// print entries specified by indices
	// keep in mind that matrices are stored
	// in column major format
	print_matrixC(mat, 0, 900);

	// print entries specified by coordinates
	size_t start[MAX_DIMS] = {0, 0};
	size_t end[MAX_DIMS] = {132, 1};
	// C2I takes coordinate array and the
	// matrix dims and converts it to an index
	print_matrixC(mat, C2I(start, mat->dims), C2I(end, mat->dims));
	
	// free matrix
	free_matrixC(mat);
}

void test_new_matrixC_device() {
	// define dimensions of matrix
	// only need to specify the dimensions we need, up to MAX_DIMS
	size_t dims[MAX_DIMS] = {768, 600};

	// allocate a new device matrix with those dimensions
	matrixC * mat = new_matrixC(dims, device); 
	
	// initialize matrix data
	for (size_t i = 0; i < mat->num; i++) {
		mat->data[i] = make_cuDoubleComplex((double)i, (double)(i*2));
	}

	// print entries specified by indices
	// keep in mind that matrices are stored
	// in column major format
	print_matrixC(mat, 0, 900);

	// print entries specified by coordinates
	size_t start[MAX_DIMS] = {0, 0};
	size_t end[MAX_DIMS] = {132, 1};
	// C2I takes coordinate array and the
	// matrix dims and converts it to an index
	print_matrixC(mat, C2I(start, mat->dims), C2I(end, mat->dims));
	
	// free matrix
	free_matrixC(mat);
}

void test_toDeviceC() {
	// define dimensions of matrix
	// only need to specify the dimensions we need, up to MAX_DIMS
	size_t dims[MAX_DIMS] = {768, 600};

	// allocate a new device matrix with those dimensions
	matrixC * mat = new_matrixC(dims, host); 
	
	// initialize matrix data
	for (size_t i = 0; i < mat->num; i++) {
		mat->data[i] = make_cuDoubleComplex((double)i, (double)(i*2));
	}

	// copy matrix to device
	// the host version is preserved
	matrixC * mat_d = toDeviceC(mat);
	
	// print entries
	// matrix is first copied to device,
	// so better to use hosts copy
	print_matrixC(mat_d, 0, 900);
	
	// free device matrix
	free_matrixC(mat_d);

	// free host matrix (because it isn't deleted when sent to device)	
	free_matrixC(mat);
}

void test_crop_matrix() {
	// define dimensions of matrix
	// only need to specify the dimensions we need, up to MAX_DIMS
	size_t dims[MAX_DIMS] = {10, 5};

	// allocate a new host matrix with those dimensions
	matrix * mat = new_matrix(dims, host); 
	
	// initialize matrix data
	for (size_t i = 0; i < mat->num; i++) {
		mat->data[i] = 	(double)i;
	}

	// print matrix before crop
	print_matrix(mat, 0, mat->num);

	// crop matrix
	// the old data is automatically freed
	size_t newDims[MAX_DIMS] = {5, 3};	
	mat = crop_matrix(mat, newDims);
	
	// print entries specified by indices
	// keep in mind that matrices are stored
	// in column major format
	print_matrix(mat, 0, mat->num);
}


void test_crop_matrixC() {
	// define dimensions of matrix
	// only need to specify the dimensions we need, up to MAX_DIMS
	size_t dims[MAX_DIMS] = {10, 5};

	// allocate a new host matrix with those dimensions
	matrixC * mat = new_matrixC(dims, host); 
	
	// initialize matrix data
	for (size_t i = 0; i < mat->num; i++) {
		mat->data[i] = make_cuDoubleComplex(
				(double)i,
				(double)(i*2));
	}

	// print matrix before crop
	print_matrixC(mat, 0, 50);

	// crop matrix
	// the old data is automatically freed
	size_t newDims[MAX_DIMS] = {5, 3};	
	mat = crop_matrixC(mat, newDims);
	
	// print entries specified by indices
	// keep in mind that matrices are stored
	// in column major format
	print_matrixC(mat, 0, 14);
}


int main(int argc, char **argv) {
	if(argc != 2) {
		fprintf(stderr, "Usage: %s ARG\n", argv[0]);
		exit(1);
	}
	if (strcmp(argv[1], "newHostC") == 0) {
		test_new_matrixC_host();
	} else if (strcmp(argv[1], "newDeviceC") == 0) {
		test_new_matrixC_device();
	} else if (strcmp(argv[1], "toDeviceC") == 0) {
		test_toDeviceC();	
	} else if (strcmp(argv[1], "crop") == 0) {
		test_crop_matrix();
	} else if (strcmp(argv[1], "cropC") == 0) {
		test_crop_matrixC();
	} else {
		printf("Not a valid command line argument\n");
	}
}

