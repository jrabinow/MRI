/*
 * Matrix types
 * 
 * 1-4 dimensions
 * double or cuDoubleComplex
 * 
 * Matrices are allocated automatically but must be freed
 * manually with free_matrix or free_matrixC
 *
 * To copy metadata without changing data, use struct assignment:
 *   matrix * mat = new_matrix(dims, host);
 *   mat2 = mat;
 *
 * TODO: malloc error checking (cuda error checking done)
 *
 */

#ifndef MATRIX
#define MATRIX

#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <cuComplex.h>
#include "cudaErr.h"

#define MAX_DIMS 4

// location flag says where data is stored
typedef enum { device, host } locationFlag;

/*
 * Internal utility function to change dimensions with size 0 to have size 1
 * oldDims and newDims should be arrays of size MAX_DIMS
 * Basically, all matrices have MAX_DIMS dimensions, but if dimension is
 * size 1 we don't worry about it. This function let's us specify only the
 * dimensions we need using array initialization, e.g. if MAX_DIMS = 4 then:
 *   size_t dims[MAX_DIMS] = {4, 20}
 * produces the array {4, 20, 0, 0}
 * As an added bonus, this also makes a copy of dims so we don't
 * have to worry about wrecking the caller's version
 */
inline void processDims(size_t * newDims, size_t * oldDims) {
	for (int i = 0; i < MAX_DIMS; i++) {
		newDims[i] = (oldDims[i] == 0) ? 1 : oldDims[i];
	}
}


/*
 * Inline functions to convert between index and coordinate
 * 
 * Both use the following:
 *   let dims[MAX_DIMS] be the dimensions of the array
 *   let p[MAX_DIMS] be a point in the array
 *   let i be the index of p
 * Then the following relates p and i (this is latex):
 *   $i = \sum_{n=0}^{n=MAX\_DIMS} (p[n] * \prod_{m=0}^{m=n-1} (dims[m]))$
 * NOTE: how well will these be optimized if we make dims constant?
 */

// multidimensional coordinate to 1 dimensional row-major index
inline size_t C2I(size_t * coord, size_t * dims) {
	size_t offset = 1;
	size_t idx = 0;
	for (int i = 0; i < MAX_DIMS; i++) {
		idx += coord[i] * offset;
		offset *= dims[i];
	}
	return idx;
}

// 1 dimensional row-major index to multidimensional coordinate
// The return value is declared static, so subsequent calls will
// overwrite previous return values. Therefore, not for use on device
inline size_t * I2C(size_t idx, size_t * dims) {
	size_t offset[MAX_DIMS];
	static size_t point[MAX_DIMS];

	// change 0's to 1's
	processDims(dims, dims);

	// compute offset multiplier when moving up the ith dimension
	offset[0] = 1;
	for (int i = 1; i < MAX_DIMS; i++) {
		offset[i] = dims[i-1] * offset[i-1];
	}

	// divide by offsets, largest first
	// (it's kind of like positional notation)
	for (int i = MAX_DIMS - 1; i >= 0; i--) {
		point[i] = idx / offset[i];
		idx = idx % offset[i];
	}

	return point;
}

/*
 * 
 * Double Matrix
 *
 */

typedef struct {
	double * data; // the data array
	locationFlag location; // location where the data is stored
	size_t num; // number of entries
	size_t size; // size in bytes of each entry
	size_t dims[MAX_DIMS]; // dimension size array
} matrix;

// constructor
matrix * new_matrix(size_t * dims, locationFlag location) {
	// allocate metadata struct
	matrix * mat = (matrix *)malloc(sizeof(matrix));

	// change dims with size 0 to have size 1
	// making copy of dims in the process
	processDims(mat->dims, dims);

	// compute number of entries
	mat->num = 1;
	for (int i = 0; i < MAX_DIMS; i++) {
		// compute num entries
		mat->num *= mat->dims[i];
	}

	// set data size and location
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

// deconstructor
void free_matrix(matrix * in) {
	// free data on host or device
	if (in->location == host) {
		free(in->data);
	} else if (in->location == device) {
		cudaFree(in->data);
	} else {
		// error
	}
	// free metadata on host
	free(in);
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
	matrix * out;
	if (in->location == device) {
		out = new_matrix(in->dims, host);
		cudaErrChk(cudaMemcpy(out->data,
				in->data,
				in->num*in->size,
				cudaMemcpyDeviceToHost));
	} else {
		// error: matrix not on device
		out = NULL;
	}
	return out;
}

// copy host matrix to device
matrix * toDevice(matrix * in) {
	matrix * out;
	if (in->location == host) {
		out = new_matrix(in->dims, device);
		cudaErrChk(cudaMemcpy(out->data,
				in->data,
				in->num*in->size,
				cudaMemcpyHostToDevice));
	} else {
		// error: matrix not on host
		out = NULL;
	}
	return out;
}

// Crop matrix
// Destroys input metadata, might reuse input data

matrix * crop_matrix(matrix * in, size_t * newDims) {
	matrix * out;
	
	// check to see if only the last dim is cropped
	bool onlyLastChanged = true;
	for (int i = 0; i < MAX_DIMS - 1; i++) {
		if (in->dims[i] != newDims[i]) {
			onlyLastChanged = false;
		}
	}

	// if so, we can just change the metadata and realloc data to smaller size
	if (onlyLastChanged) {
		out = in;
		out->num = 1;
		for (int i = 0; i < MAX_DIMS; i++) {
			out->dims[i] = newDims[i];
			out->num *= out->dims[i];
		}
		out->data = (double *)realloc(out->data, out->num*out->size);
	} else {
		// otherwise, we have to actually rearrange the data

		// create output matrix
		if (in->location == host) {
			out = new_matrix(newDims, host);
		} else if (in->location == device) {
			out = new_matrix(newDims, device);
		} else {
			// error
		}

		// loop over the beginnings of the columns of the output matrix
		size_t * out_coord;
		size_t in_idx;
		for (size_t i = 0; i < out->num; i += out->dims[0]) {
			// convert index relative to out matrix
			// to index relative to input matrix
			out_coord = I2C(i, out->dims);
			in_idx = C2I(out_coord, in->dims);
			
			// copy the desired portion of this column
			if (in->location == host) {
				memcpy(&(out->data[i]),
						&(in->data[in_idx]),
						out->dims[0]*out->size);
			} else if (in->location == device) {
				cudaErrChk(cudaMemcpy(&(out->data[i]),
						&(in->data[in_idx]),
						out->dims[0],
						cudaMemcpyDeviceToDevice));
			} else {
				// error
			}
		}
		// free input data
		free_matrix(in);	
	}
	return out;
}

// print matrix from start index to end index
void print_matrix(matrix * in, size_t start, size_t end) {
	// if matrix is on device, copy it to host
	bool usingCopy = false;
	if (in->location == device) {
		in = toHost(in);
		usingCopy = true;
	}

	// print matrix entries
	size_t * coord;
	size_t firstCoord;
	for (size_t i = start; i < end; i++) {
		// if entry is the start of a column, print header
		coord = I2C(i, in->dims);
		firstCoord = coord[0];
		if (firstCoord == 0) {
			printf("\nColumn %d:\n\n", I2C(i, in->dims)[1]); 
		}
		// print entry
		printf("%f\n", in->data[i]);
	}

	// if we copied to host, free our copy
	if (usingCopy) {
		free_matrix(in);
	}
}




/*
 *
 * cuDoubleComplex Matrix
 *
 */

typedef struct {
	cuDoubleComplex * data; // the data array
	locationFlag location; // location where the data is stored
	size_t num; // number of entries
	size_t size; // size in bytes of each entry
	size_t dims[MAX_DIMS]; // dimension size array
} matrixC;

// constructor
matrixC * new_matrixC(size_t * dims, locationFlag location) {
	// allocate metadata struct
	matrixC * mat = (matrixC *)malloc(sizeof(matrixC));

	// change dims with size 0 to have size 1
	// making copy of dims in the process
	processDims(mat->dims, dims);

	// compute number of entries
	mat->num = 1;
	for (int i = 0; i < MAX_DIMS; i++) {
		// compute num entries
		mat->num *= mat->dims[i];
	}

	// set data size and location
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

// deconstructor
void free_matrixC(matrixC * in) {
	// free data on host or device
	if (in->location == host) {
		free(in->data);
	} else if (in->location == device) {
		cudaFree(in->data);
	} else {
		// error
	}
	// free metadata on host
	free(in);
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
	matrixC * out;
	if (in->location == device) {
		out = new_matrixC(in->dims, host);
		cudaErrChk(cudaMemcpy(out->data,
				in->data,
				in->num*in->size,
				cudaMemcpyDeviceToHost));
	} else {
		// error: matrix not on device
		out = NULL;
	}
	return out;
}

// copy host matrix to device
matrixC * toDeviceC(matrixC * in) {
	matrixC * out;
	if (in->location == host) {
		out = new_matrixC(in->dims, device);
		cudaErrChk(cudaMemcpy(out->data,
				in->data,
				in->num*in->size,
				cudaMemcpyHostToDevice));
	} else {
		// error: matrix not on host
		out = NULL;
	}
	return out;
}

// Crop matrix
// Destroys input metadata, might reuse input data

matrixC * crop_matrixC(matrixC * in, size_t * newDims) {
	matrixC * out;
	
	// check to see if only the last dim is cropped
	bool onlyLastChanged = true;
	for (int i = 0; i < MAX_DIMS - 1; i++) {
		if (in->dims[i] != newDims[i]) {
			onlyLastChanged = false;
		}
	}

	// if so, we can just change the metadata and realloc data to smaller size
	if (onlyLastChanged) {
		out = in;
		out->num = 1;
		for (int i = 0; i < MAX_DIMS; i++) {
			out->dims[i] = newDims[i];
			out->num *= out->dims[i];
		}
		out->data = (cuDoubleComplex *)realloc(out->data, out->num*out->size);
	} else {
		// otherwise, we have to actually rearrange the data

		// create output matrix
		if (in->location == host) {
			out = new_matrixC(newDims, host);
		} else if (in->location == device) {
			out = new_matrixC(newDims, device);
		} else {
			// error
		}

		// loop over the beginnings of the columns of the output matrix
		size_t * out_coord;
		size_t in_idx;
		for (size_t i = 0; i < out->num; i += out->dims[0]) {
			// convert index relative to out matrix
			// to index relative to input matrix
			out_coord = I2C(i, out->dims);
			in_idx = C2I(out_coord, in->dims);
			
			// copy the desired portion of this column
			if (in->location == host) {
				memcpy(&(out->data[i]),
						&(in->data[in_idx]),
						out->dims[0]*out->size);
			} else if (in->location == device) {
				cudaErrChk(cudaMemcpy(&(out->data[i]),
						&(in->data[in_idx]),
						out->dims[0],
						cudaMemcpyDeviceToDevice));
			} else {
				// error
			}
		}
		// free input data
		free_matrixC(in);	
	}
	return out;
}



// print matrix from start index to end index
void print_matrixC(matrixC * in, size_t start, size_t end) {
	// if matrix is on device, copy it to host
	bool usingCopy = false;
	if (in->location == device) {
		in = toHostC(in);
		usingCopy = true;
	}

	// print matrix entries
	size_t * coord;
	size_t firstCoord;
	for (size_t i = start; i < end; i++) {
		// if entry is the start of a column, print header
		coord = I2C(i, in->dims);
		firstCoord = coord[0];
		if (firstCoord == 0) {
			printf("\nColumn %d:\n\n", I2C(i, in->dims)[1]); 
		}
		// print entry
		printf("%f + %fi\n",
				cuCreal(in->data[i]),
				cuCimag(in->data[i]));
	}

	// if we copied to host, free our copy
	if (usingCopy) {
		free_matrixC(in);
	}
}


#endif
