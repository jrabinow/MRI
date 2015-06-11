/*
 * Matrix type
 *
 * 1-4 dimensions
 * double or cuDoubleComplex
 *
 * Matrices are allocated automatically but must be freed
 * manually with delete_Matrix
 *
 * To copy metadata without changing data, use struct assignment:
 *   Matrix* mat = new_matrix(dims, HOST, TYPE);
 *   mat2 = mat;
 */
#ifndef MATRIX_H
#define MATRIX_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <cuComplex.h>
#include <cudaErr.h>
#include <utils.h>

#define MAX_DIMS 4

// location flag says where data is stored
typedef enum { DEVICE, HOST } locationFlag;
/* var flag says which type of data is contained in the matrix */
typedef enum { DOUBLE, COMPLEX } varFlag;

/* Matrix type */
typedef struct {
	__extension__
	union {
		void *data;
		double *ddata;
		cuDoubleComplex *cdata;
	};
	varFlag vartype;
	locationFlag location; 	// location where the data is stored
	size_t num;		// number of entries
	size_t size;		// size in bytes of each entry
	size_t dims[MAX_DIMS];	// dimension size array
#ifdef DEBUG
	unsigned mat_id; /* assign a unique ID to each matrix for debugging */
#endif
} Matrix;

Matrix* new_Matrix(size_t* dims, locationFlag location, varFlag vartype);
void delete_Matrix(Matrix* in);
Matrix* copy(Matrix* in);
Matrix* toHost(Matrix* in);
Matrix* toDevice(Matrix* in);
Matrix * crop_Matrix(Matrix* in, size_t* newDims);
void print_Matrix(Matrix* in, size_t start, size_t end);

/* Inline functions go in header files */
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
inline void processDims(size_t* newDims, size_t* oldDims)
{
	int i;
	for (i = 0; i < MAX_DIMS; i++) {
		newDims[i] = oldDims[i] + 1 - (oldDims[i] != 0);
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
 *
 * TODO: how well will these be optimized if we make dims constant?
 *
 * Optimized enough. However, it's possible to write this without a for loop.
 * How often is this thing going to be called?
 * If it's only once in a while, I guess we can leave it as is. If not, we
 * should definitely consider writing something like in the CUBLAS manual
 * (section 1.1).
 */
// multidimensional coordinate to 1 dimensional row-major index
inline size_t C2I(size_t* coord, size_t* dims)
{
	size_t offset = 1;
	size_t idx = 0;
	int i;

	for (i = 0; i < MAX_DIMS; i++) {
		idx += coord[i] * offset;
		offset *= dims[i];
	}
	return idx;
}

/* 1 dimensional row-major index to multidimensional coordinate
 * TODO: updated so return value is no longer static. This can be used on
 * the device
 * Not sure what happens when you combine static and inline anyways. Probably
 * best not to find out, C++ is weird enough as it is.
 */
typedef struct {
	size_t coord[MAX_DIMS];
} Coordinate;

inline Coordinate I2C(size_t idx, size_t* dims)
{
	size_t offset[MAX_DIMS];
	Coordinate point;
	int i;

	// change 0's to 1's
	processDims(dims, dims);

	// compute offset multiplier when moving up the ith dimension
	offset[0] = 1;
	for (i = 1; i < MAX_DIMS; i++) {
		offset[i] = dims[i-1] * offset[i-1];
	}

	// divide by offsets, largest first
	// (it's kind of like positional notation)
	for (i = MAX_DIMS - 1; i >= 0; i--) {
		point.coord[i] = idx / offset[i];
		idx = idx % offset[i];
	}

	return point;
}


#ifdef __cplusplus
}
#endif
#endif
