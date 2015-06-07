/*
		printf("%f + %fi\n",
				cuCreal(in->data[i]),
				cuCimag(in->data[i]));
 * Matrix types
 *
 * 1-4 dimensions
 * double or cuDoubleComplex
 *
 * Matrices are allocated automatically but must be freed
 * manually with delete_matrix or delete_matrixC
 *
 * To copy metadata without changing data, use struct assignment:
 *   matrix * mat = new_matrix(dims, HOST, TYPE);
 *   mat2 = mat;
 */

#ifndef MATRIX
#define MATRIX

#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <cuComplex.h>
#include "cudaErr.h"
#include "utils.h"

#define MAX_DIMS 4

// location flag says where data is stored
typedef enum { DEVICE, HOST } locationFlag;
typedef enum { DOUBLE, COMPLEX } varFlag;

/*
 * Internal utility function to change dimensions with size 0 to have size 1
 * oldDims and newDims should be arrays of size MAX_DIMS
 * Basically, all matrices have MAX_DIMS dimensions, but if dimension is
 * size 1 we don't worry about it. This function lets us specify only the
 * dimensions we need using array initialization, e.g. if MAX_DIMS = 4 then:
 *   size_t dims[MAX_DIMS] = {4, 20}
 * produces the array {4, 20, 0, 0}
 * As an added bonus, this also makes a copy of dims so we don't
 * have to worry about wrecking the caller's version
 */
inline void processDims(size_t * newDims, size_t * oldDims) {
	for (int i = 0; i < MAX_DIMS; i++) {
		newDims[i] = oldDims[i] + 1 - (oldDims[i] != 0);
		/* equivalent to
		 * newDims[i] = (oldDims[i] == 0) ? 1 : oldDims[i]; */
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

/* Matrix type */
typedef struct {
	__extension__
	union {
		void *data;
		double *ddata;
		cuDoubleComplex *cdata;
	};
	varFlag vartype;
	locationFlag location; // location where the data is stored
	size_t num; // number of entries
	size_t size; // size in bytes of each entry
	size_t dims[MAX_DIMS]; // dimension size array
#ifdef DEBUG
	unsigned mat_id;
#endif
} Matrix;

// constructor
Matrix * new_Matrix(size_t * dims, locationFlag location, varFlag vartype) {
#ifdef DEBUG
	static unsigned mat_id = 0;
	if(dims == NULL) {
		log_message(LOG_FATAL, "NULL pointer passed in %s\n", __func__);
		exit(EXIT_FAILURE);
	}
#endif
	// allocate metadata struct
	Matrix * mat = (Matrix*) xmalloc(sizeof(Matrix));

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
	mat->vartype = vartype;
	if(vartype == DOUBLE) {
		mat->size = sizeof(double);
	} else {
		mat->size = sizeof(cuDoubleComplex);
	}
	mat->location = location;

	// allocate data array
	if (mat->location == HOST) {
		mat->data = xmalloc(mat->num * mat->size);
	} else if (mat->location == DEVICE) {
		cudaErrChk(cudaMalloc((void**)&(mat->data),
				mat->num*mat->size));
	} else {
		log_message(LOG_FATAL, "Unknown matrix location %s:%s", __func__, __LINE__);
		exit(EXIT_FAILURE);
	}
#ifdef DEBUG
	mat->mat_id = mat_id++;
#endif
	return mat;
}

// deconstructor
void delete_Matrix(Matrix * in) {
#ifdef DEBUG
	if(in == NULL) {
		log_message(LOG_FATAL, "NULL pointer passed in %s\n", __func__);
		exit(EXIT_FAILURE);
	}
	log_message(LOG_DEBUG, "matrix id = %u\n", in->mat_id);
#endif
	// free data on host or device
	if (in->location == HOST) {
#ifdef DEBUG
		log_message(LOG_DEBUG, "%s\tFreeing in->data", __func__);
#endif
		free(in->data);
	} else if (in->location == DEVICE) {
#ifdef DEBUG
		log_message(LOG_DEBUG, "%s\tCUDAFreeing in->data", __func__);
#endif
		cudaFree(in->data);
	} else {
		log_message(LOG_FATAL, "Unknown matrix location %s:%s", __func__, __LINE__);
		exit(EXIT_FAILURE);
	}
	// free metadata on host
#ifdef DEBUG
	log_message(LOG_DEBUG, "%s\tFreeing in", __func__);
#endif
	free(in);
}

// copy matrix maintaining location
Matrix * copy(Matrix * in) {
#ifdef DEBUG
	if(in == NULL) {
		log_message(LOG_FATAL, "NULL pointer passed in %s\n", __func__);
		exit(EXIT_FAILURE);
	}
#endif
	Matrix * out = new_Matrix(in->dims, in->location, in->vartype);
	if (in->location == HOST) {
		memcpy(out->data, in->data, in->num * in->size);
	} else if (in->location == DEVICE) {
		cudaErrChk(cudaMemcpy(out->data,
				in->data,
				in->num * in->size,
				cudaMemcpyDeviceToDevice));
	} else {
		log_message(LOG_FATAL, "Unknown matrix location %s:%s", __func__, __LINE__);
		exit(EXIT_FAILURE);
	}
	return out;
}

// copy device matrix to host
Matrix * toHost(Matrix * in) {
#ifdef DEBUG
	if(in == NULL) {
		log_message(LOG_FATAL, "NULL pointer passed in %s\n", __func__);
		exit(EXIT_FAILURE);
	}
#endif
	Matrix * out = NULL;
	if (in->location == DEVICE) {
		out = new_Matrix(in->dims, HOST, in->vartype);
		cudaErrChk(cudaMemcpy(out->data, in->data,
			in->num * in->size, cudaMemcpyDeviceToHost));
	}
	return out;
}

// copy host matrix to device
Matrix *toDevice(Matrix * in) {
#ifdef DEBUG
	if(in == NULL) {
		log_message(LOG_FATAL, "NULL pointer passed in %s\n", __func__);
		exit(EXIT_FAILURE);
	}
#endif
	Matrix * out = NULL;
	if (in->location == HOST) {
		out = new_Matrix(in->dims, DEVICE, in->vartype);
		cudaErrChk(cudaMemcpy(out->data, in->data,
			in->num*in->size, cudaMemcpyHostToDevice));
	}
	return out;
}

// Crop matrix
// Destroys input metadata, might reuse input data

Matrix * crop_Matrix(Matrix * in, size_t * newDims) {
#ifdef DEBUG
	if(in == NULL || newDims == NULL) {
		log_message(LOG_FATAL, "NULL pointer passed in %s\n", __func__);
		exit(EXIT_FAILURE);
	}
#endif
	Matrix * out = NULL;

	// check to see if only the last dim is cropped
	bool onlyLastChanged = true;
	for (int i = 0; i < MAX_DIMS - 1; i++) {
		if (in->dims[i] != newDims[i]) {
			onlyLastChanged = false;
			break;
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
		out->data = (double*) xrealloc(out->data, out->num * out->size);
	} else {
		// otherwise, we have to actually rearrange the data
		// create output matrix
		if (in->location == HOST) {
			out = new_Matrix(newDims, HOST, in->vartype);
		} else if (in->location == DEVICE) {
			out = new_Matrix(newDims, DEVICE, in->vartype);
		} else {
			log_message(LOG_FATAL, "Unknown matrix location %s:%s", __func__, __LINE__);
			exit(EXIT_FAILURE);
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
			if (in->location == HOST) {
				if(in->vartype == DOUBLE)
					memcpy(&(out->ddata[i]), &(in->ddata[in_idx]),
							out->dims[0]*out->size);
				else
					memcpy(&(out->cdata[i]), &(in->cdata[in_idx]),
							out->dims[0]*out->size);
			} else if (in->location == DEVICE) {
				if(in->vartype == DOUBLE) {
					cudaErrChk(cudaMemcpy(&(out->ddata[i]),
						&(in->ddata[in_idx]),
						out->dims[0],
						cudaMemcpyDeviceToDevice));
				} else {
					cudaErrChk(cudaMemcpy(&(out->cdata[i]),
						&(in->cdata[in_idx]),
						out->dims[0],
						cudaMemcpyDeviceToDevice));
				}
			} else {
				log_message(LOG_FATAL, "Unknown matrix location %s:%s", __func__, __LINE__);
				exit(EXIT_FAILURE);
			}
		}
		// free input data
		delete_Matrix(in);
	}
	return out;
}

// print matrix from start index to end index
void print_Matrix(Matrix * in, size_t start, size_t end) {
#ifdef DEBUG
	if(in == NULL) {
		log_message(LOG_FATAL, "NULL pointer passed in %s\n", __func__);
		exit(EXIT_FAILURE);
	}
#endif
	// if matrix is on device, copy it to host
	bool usingCopy = false;
	if (in->location == DEVICE) {
		in = toHost(in);
		usingCopy = true;
	}

	// print matrix entries
	size_t * coord;
	size_t firstCoord;
	if(in->vartype == DOUBLE)
		for (size_t i = start; i < end; i++) {
			// if entry is the start of a column, print header
			coord = I2C(i, in->dims);
			firstCoord = coord[0];
			if (firstCoord == 0) {
				printf("\nColumn %d:\n\n", I2C(i, in->dims)[1]);
			}
			// print entry
			printf("%f\n", in->ddata[i]);
		}
	else
		for (size_t i = start; i < end; i++) {
			// if entry is the start of a column, print header
			coord = I2C(i, in->dims);
			firstCoord = coord[0];
			if (firstCoord == 0) {
				printf("\nColumn %d:\n\n", I2C(i, in->dims)[1]);
			}
			// print entry
			printf("%f + %fi\n", cuCreal(in->cdata[i]), cuCimag(in->cdata[i]));
		}

	// if we copied to host, free our copy
	if (usingCopy) {
		delete_Matrix(in);
	}
}

#endif
