/*********************************/
/* Matrix types                  */
/* 1-4 dimensions                */
/* double or cuDoubleComplex     */
/*********************************/

#ifndef MATRIX
#define MATRIX

#include <string.h>
#include <complex.h>

#define MAX_DIMS 4

// location flag says where data is stored
typedef enum { device, host } locationFlag;


/***************************************************************************/
/* Inline functions to convert between index and coordinate                */
/* Both make use the following:                                            */
/*   let dims[MAX_DIMS] be the dimensions of the array                     */
/*   let p[MAX_DIMS] be a point in the array                               */
/*   let i be the index of p                                               */
/* Then (this is latex):                                                   */
/*   $i = \sum_{n=0}^{n=MAX\_DIMS} (p[n] * \prod_{m=0}^{m=n-1} (dims[m]))$ */
/* NOTE: how well will these be optimized if we make dims constant?        */
/***************************************************************************/

// multidimensional coordinate to 1 dimensional row-major index
inline int C2I(int[] coord, int[] dims) {
	int i;
	int offset = 1;
	int idx = 0;
	for (i = 0; i < MAX_DIMS; i++) {
		idx += point[i] * offset;
		offset *= dims[i];
	}
	return idx;
}

// 1 dimensional row-major index to multidimensional coordinate
inline int[] I2C(int idx, int[] dims) {
	int i;
	int offset[MAX_DIMS];
	int point[MAX_DIMS];
	// compute offset multiplier when moving up the ith dimension
	for (i = 1; i < MAX_DIMS; i++) {
		offset[i] = dims[i] * offset[i-1];
	}
	// divide by offsets, largest first
	// (it's kind of like positional notation)
	for (i = MAX_DIMS; i > 0 ; i--) {
		point[i] = idx / offset;
		idx = idx % dims[i];
	}
	return point;
}




/**********************/
/* Double Matrix      */
/**********************/

typedef struct {
	int num; // number of entries
	int dims[MAX_DIMS]; // dimension size array
	size_t size; // size in bytes of each entry
	locationFlag location; // location where the data is stored
	double * data; // the data array
} matrix;

// constructor
matrix * new_matrix(int dims[], locationFlag location) {
	int i;
	// allocate metadata struct
	matrix * mat = (matrix *)malloc(sizeof(matrix));
	// copy dimension array
	// and add up number of entries at the same time
	mat->num = 1;
	for (i = 0; i < MAX_DIMS; i++) {
		mat->dims[i] = dims[i];
		mat->num *= dims[i];
	}
	mat->size = sizeof(double);
	mat->location = location;
	// allocate data array
	if (mat->location == host) {
		mat->data = (double *)malloc(mat->num*mat->size);
	} else if (mat->location == device) {
		cudaErrChk(cudaMalloc((void**)&(mat->data),
				mat->num*mat->size));
	} else {
		// error
	}
	return mat;
}

// copy matrix maintaining location
matrix * copy(matrix * in) {
	matrix * out = new_matrix(in->dims, in->location);
	if (in->location == host) {
		memcpy(out->data, in->data, in->num*in->size);
	} else if (in->location == device) {
		cudaErrChk(cudaMemcpy(out->data,
				in->data,
				in->num*in->size,
				cudaMemcpyDeviceToDevice));
	} else {
		// error
	}
	return out;
}

// copy device matrix to host
matrix * toHost(matrix * in) {
	if (in->location == device) {
		matrix * out = new_matrix(in->dims, host);
		cudaErrChk(cudaMemcpy(out->data,
				in->data,
				in->num*in->size,
				cudaMemcpyDeviceToHost));
	} else {
		// error: matrix not on device
	}
	return out;
}

// copy host matrix to device
matrix * toDevice(matrix * in) {
	if (in->location == host) {
		matrix * out = new_matrix(in->dims, device);
		cudaErrChk(cudaMemcpy(out->data,
				in->data,
				in->num*in->size,
				cudaMemcpyDeviceToHost));
	} else {
		// error: matrix not on host
	}
	return out;
}

// print rectangular selection entry by entry, lower dimensions first
void print_matrix(matrix * in, int start[], int end[]) {
	int i, j;
	int select_dims[MAX_DIMS];
	int select_num = 1;

	// if matrix is on device, copy it to host
	if (in->location == device) {
		in = toHost(in);
	}

	// compute selection dimensions and number of entries
	for (i = 0; i < MAX_DIMS; i++) {
		select_dims[i] = end[i] - start[i];
		select_num *= select_dims[i];
	}

	// We loop over indices relative to the selection
	// (at first ignoring the indices the selection inherits
	// from the containing matrix).
	// We convert each index to a coordinate relative
	// to the selection, then shift by the start coordinate
	// to get a coordinate relative to the containing matrix.
	// Finally, we convert this coordinate to an index
	// relative to the containing matrix
	for (i = 0; i < select_num; i++) {
		int contain_coord[MAX_DIMS];
		int contain_index;
		// convert index to coordinate relative to selection
		int select_coord[] = I2C(i, select_dims)
		// shift by start coordinate to get
		// coordinate relative to containing matrix
		for (j = 0; j < MAX_DIMS; j++) {
			contain_coord[j] = start[j] + select_coord[j];
		}
		// convert to index relative to containing matrix
		contain_index = C2I(contain_coord);
		// if this is the first entry in a column, print a header
		// NOTE: this depends on MAX_DIMS = 4
		if (select_coord[0] == start[0]) {
			printf("Column (%i:%i, %i, %i, %i)",
					start[0],
					end[0],
					contain_coord[1],
					contain_coord[2],
					contain_coord[3]);
		}
		// print entry
		printf("%f\n", mat->data[contain_index]);
	}
}




/*******************************/
/* cuDoubleComplex Matrix      */
/*******************************/

typedef struct {
	int num; // number of entries
	int dims[MAX_DIMS]; // dimension size array
	size_t size; // size in bytes of each entry
	locationFlag location; // location where the data is stored
	cuDoubleComplex * data; // the data array
} matrixC;

// constructor
matrixC * new_matrixC(int dims[], locationFlag location) {
	int i;
	// allocate metadata struct
	matrixC * mat = (matrixC *)malloc(sizeof(matrixC));
	// copy dimension array
	// and add up number of entries at the same time
	mat->num = 1;
	for (i = 0; i < MAX_DIMS; i++) {
		mat->dims[i] = dims[i];
		mat->num *= dims[i];
	}
	mat->size = sizeof(cuDoubleComplex);
	mat->location = location;
	// allocate data array
	if (mat->location == host) {
		mat->data = (cuDoubleComplex *)malloc(mat->num*mat->size);
	} else if (mat->location == device) {
		cudaErrChk(cudaMalloc((void**)&(mat->data),
				mat->num*mat->size));
	} else {
		// error
	}
	return mat;
}

// copy matrix maintaining location
matrixC * copyC(matrixC * in) {
	matrixC * out = new_matrixC(in->dims, in->location);
	if (in->location == host) {
		memcpy(out->data, in->data, in->num*in->size);
	} else if (in->location == device) {
		cudaErrChk(cudaMemcpy(out->data,
				in->data,
				in->num*in->size,
				cudaMemcpyDeviceToDevice));
	} else {
		// error
	}
	return out;
}

// copy device matrix to host
matrixC * toHostC(matrixC * in) {
	if (in->location == device) {
		matrixC * out = new_matrixC(in->dims, host);
		cudaErrChk(cudaMemcpy(out->data,
				in->data,
				in->num*in->size,
				cudaMemcpyDeviceToHost));
	} else {
		// error: matrix not on device
	}
	return out;
}

// copy host matrix to device
matrixC * toDeviceC(matrixC * in) {
	if (in->location == host) {
		matrixC * out = new_matrixC(in->dims, device);
		cudaErrChk(cudaMemcpy(out->data,
				in->data,
				in->num*in->size,
				cudaMemcpyDeviceToHost));
	} else {
		// error: matrix not on host
	}
	return out;
}

// print rectangular selection entry by entry, lower dimensions first
void print_matrixC(matrixC * in, int start[], int end[]) {
	int i, j;
	int select_dims[MAX_DIMS];
	int select_num = 1;

	// if matrix is on device, copy it to host
	if (in->location == device) {
		in = toHostC(in);
	}

	// compute selection dimensions and number of entries
	for (i = 0; i < MAX_DIMS; i++) {
		select_dims[i] = end[i] - start[i];
		select_num *= select_dims[i];
	}

	// We loop over indices relative to the selection
	// (at first ignoring the indices the selection inherits
	// from the containing matrix).
	// We convert each index to a coordinate relative
	// to the selection, then shift by the start coordinate
	// to get a coordinate relative to the containing matrix.
	// Finally, we convert this coordinate to an index
	// relative to the containing matrix
	for (i = 0; i < select_num; i++) {
		int contain_coord[MAX_DIMS];
		int contain_index;
		double real, imag;
		// convert index to coordinate relative to selection
		int select_coord[] = I2C(i, select_dims)
		// shift by start coordinate to get
		// coordinate relative to containing matrix
		for (j = 0; j < MAX_DIMS; j++) {
			contain_coord[j] = start[j] + select_coord[j];
		}
		// convert to index relative to containing matrix
		mat_index = C2I(mat_coord);
		// if this is the first entry in a column, print a header
		// NOTE: this depends on MAX_DIMS = 4
		if (select_coord[0] == start[0]) {
			printf("Column (%i:%i, %i, %i, %i)",
					start[0],
					end[0],
					contain_coord[1],
					contain_coord[2],
					contain_coord[3]);
		}
		// separate real and imaginary parts and print
		real = cuCreal(mat->data[mat_index]);
		imag = cuCimag(mat->data[mat_index]);
		printf("%f + %fi\n", real, imag);
	}
}


#endif
