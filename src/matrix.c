/*
 * Matrix types
 *
 * 1-4 dimensions
 * double or cuDoubleComplex
 *
 * Matrices are allocated automatically but must be freed
 * manually with delete_matrix or delete_matrixC
 *
 * To copy metadata without changing data, use struct assignment:
 *   Matrix* mat = new_matrix(dims, HOST, TYPE);
 *   mat2 = mat;
 */

#include <matrix.h>

// constructor
Matrix* new_Matrix(size_t* dims, locationFlag location, varFlag vartype)
{
	Matrix* mat = NULL;
	int i;
#ifdef DEBUG
	static unsigned mat_id = 0;
	if(dims == NULL) {
		log_message(LOG_FATAL, "NULL pointer passed in %s\n", __func__);
		exit(EXIT_FAILURE);
	}
#endif
	mat = (Matrix*) xmalloc(sizeof(Matrix));

	// change dims with size 0 to have size 1
	// making copy of dims in the process
	processDims(mat->dims, dims);

	// compute number of entries
	mat->num = 1;
	for (i = 0; i < MAX_DIMS; i++) {
		// compute num entries
		mat->num *= mat->dims[i];
	}

	// set data size and location
	mat->vartype = vartype;
	if(vartype == DOUBLE)
		mat->size = sizeof(double);
	else if(vartype == COMPLEX)
		mat->size = sizeof(cuDoubleComplex);
	else {
		log_message(LOG_FATAL, "Unkown matrix type %d at %s:%s\n", vartype, __func__, __LINE__);
		exit(EXIT_FAILURE);
	}

	mat->location = location;
	// allocate data array
	if (mat->location == HOST) {
		mat->data = xmalloc(mat->num * mat->size);
	} else if (mat->location == DEVICE) {
		cudaErrChk(cudaMalloc((void**)&(mat->data),
				mat->num*mat->size));
	} else {
		log_message(LOG_FATAL, "Unknown matrix location %d at %s:%s", location, __func__, __LINE__);
		exit(EXIT_FAILURE);
	}
#ifdef DEBUG
	mat->mat_id = mat_id++;
#endif
	return mat;
}

// deconstructor
void delete_Matrix(Matrix* in)
{
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
		log_message(LOG_FATAL, "Unknown matrix location %d at %s:%s", in->location, __func__, __LINE__);
		exit(EXIT_FAILURE);
	}
	// free metadata on host
#ifdef DEBUG
	log_message(LOG_DEBUG, "%s\tFreeing in", __func__);
#endif
	free(in);
}

// copy matrix maintaining location
Matrix* copy(Matrix* in)
{
	Matrix* out = NULL;
#ifdef DEBUG
	if(in == NULL) {
		log_message(LOG_FATAL, "NULL pointer passed in %s\n", __func__);
		exit(EXIT_FAILURE);
	}
#endif
	out = new_Matrix(in->dims, in->location, in->vartype);
	if (in->location == HOST) {
		memcpy(out->data, in->data, in->num * in->size);
	} else if (in->location == DEVICE) {
		cudaErrChk(cudaMemcpy(out->data,
				in->data,
				in->num * in->size,
				cudaMemcpyDeviceToDevice));
	} else {
		log_message(LOG_FATAL, "Unknown matrix location %d at %s:%s", in->location, __func__, __LINE__);
		exit(EXIT_FAILURE);
	}
	return out;
}

// copy device matrix to host
Matrix* toHost(Matrix* in)
{
	Matrix* out = NULL;
#ifdef DEBUG
	if(in == NULL) {
		log_message(LOG_FATAL, "NULL pointer passed in %s\n", __func__);
		exit(EXIT_FAILURE);
	}
#endif
	if (in->location == DEVICE) {
		out = new_Matrix(in->dims, HOST, in->vartype);
		cudaErrChk(cudaMemcpy(out->data, in->data,
			in->num * in->size, cudaMemcpyDeviceToHost));
	}
	return out;
}

// copy host matrix to device
Matrix* toDevice(Matrix* in)
{
	Matrix* out = NULL;
#ifdef DEBUG
	if(in == NULL) {
		log_message(LOG_FATAL, "NULL pointer passed in %s\n", __func__);
		exit(EXIT_FAILURE);
	}
#endif
	if (in->location == HOST) {
		out = new_Matrix(in->dims, DEVICE, in->vartype);
		cudaErrChk(cudaMemcpy(out->data, in->data,
			in->num*in->size, cudaMemcpyHostToDevice));
	}
	return out;
}

// Crop matrix
// Destroys input metadata, might reuse input data

Matrix* crop_Matrix(Matrix* in, size_t* newDims)
{
	bool onlyLastChanged = true;
	Coordinate out_coord;
	size_t in_idx;
	size_t i;
#ifdef DEBUG
	if(in == NULL || newDims == NULL) {
		log_message(LOG_FATAL, "NULL pointer passed in %s\n", __func__);
		exit(EXIT_FAILURE);
	}
#endif
	Matrix* out = NULL;

	// check to see if only the last dim is cropped
	for (i = 0; i < MAX_DIMS - 1 && in->dims[i] != 1; i++) {
		if (in->dims[i] != newDims[i]) {
			onlyLastChanged = false;
			break;
		}
	}

	// if so, we can just change the metadata and realloc data to smaller size
	if (onlyLastChanged) {
		out = in;
		out->num = 1;
		for (i = 0; i < MAX_DIMS; i++) {
			out->dims[i] = newDims[i];
			out->num *= out->dims[i];
		}
		out->data = (double*) xrealloc(out->data, out->num * out->size);
	} else {
		// otherwise, we have to actually rearrange the data
		// create output matrix
		out = new_Matrix(newDims, in->location, in->vartype);

		// loop over the beginnings of the columns of the output matrix
		for (i = 0; i < out->num; i += out->dims[0]) {
			// convert index relative to out matrix
			// to index relative to input matrix
			out_coord = I2C(i, out->dims);
			in_idx = C2I(out_coord.coord, in->dims);

			// copy the desired portion of this column
			if (in->location == HOST) {
				if(in->vartype == DOUBLE)
					memcpy(&(out->ddata[i]), &(in->ddata[in_idx]),
							out->dims[0]*out->size);
				else if(in->vartype == COMPLEX)
					memcpy(&(out->cdata[i]), &(in->cdata[in_idx]),
							out->dims[0]*out->size);
				else {
					log_message(LOG_FATAL, "Unkown matrix type %d at %s:%s\n", in->vartype, __func__, __LINE__);
					exit(EXIT_FAILURE);
				}
			} else if (in->location == DEVICE) {
				if(in->vartype == DOUBLE) {
					cudaErrChk(cudaMemcpy(&(out->ddata[i]),
						&(in->ddata[in_idx]),
						out->dims[0],
						cudaMemcpyDeviceToDevice));
				} else if(in->vartype == COMPLEX) {
					cudaErrChk(cudaMemcpy(&(out->cdata[i]),
						&(in->cdata[in_idx]),
						out->dims[0],
						cudaMemcpyDeviceToDevice));
					memcpy(&(out->cdata[i]), &(in->cdata[in_idx]),
							out->dims[0]*out->size);
				} else {
					log_message(LOG_FATAL, "Unkown matrix type %d at %s:%s\n", in->vartype, __func__, __LINE__);
					exit(EXIT_FAILURE);
				}
			} else {
				log_message(LOG_FATAL, "Unknown matrix location %d at %s:%s", in->location, __func__, __LINE__);
				exit(EXIT_FAILURE);
			}
		}
		// free input data
		delete_Matrix(in);
	}
	return out;
}

// print matrix from start index to end index
void print_Matrix(Matrix* in, size_t start, size_t end)
{
	bool usingCopy = false;
	Coordinate out_coord;
	size_t firstCoord, i;
#ifdef DEBUG
	if(in == NULL) {
		log_message(LOG_FATAL, "NULL pointer passed in %s\n", __func__);
		exit(EXIT_FAILURE);
	}
#endif
	// if matrix is on device, copy it to host
	if (in->location == DEVICE) {
		in = toHost(in);
		usingCopy = true;
	}

	// print matrix entries
	if(in->vartype == DOUBLE)
		for (i = start; i < end; i++) {
			// if entry is the start of a column, print header
			out_coord = I2C(i, in->dims);
			firstCoord = out_coord.coord[0];
			if (firstCoord == 0) {
				printf("\nColumn %d:\n\n", I2C(i, in->dims).coord[1]);
			}
			// print entry
			printf("%f\n", in->ddata[i]);
		}
	else if(in->vartype == COMPLEX)
		for (i = start; i < end; i++) {
			// if entry is the start of a column, print header
			out_coord = I2C(i, in->dims);
			firstCoord = out_coord.coord[0];
			if (firstCoord == 0) {
				printf("\nColumn %d:\n\n", I2C(i, in->dims).coord[1]);
			}
			// print entry
			printf("%f + %fi\n", cuCreal(in->cdata[i]), cuCimag(in->cdata[i]));
		}
	else {
		log_message(LOG_FATAL, "Unkown matrix type %d at %s:%s\n", in->vartype, __func__, __LINE__);
		exit(EXIT_FAILURE);
	}

	// if we copied to host, free our copy
	if (usingCopy) {
		delete_Matrix(in);
	}
}

void dump_Matrix(const char *path, Matrix* in)
{
	bool usingCopy = false;
	Coordinate out_coord;
	size_t firstCoord, i;
	FILE *dump_file = NULL;
#ifdef DEBUG
	if(in == NULL) {
		log_message(LOG_FATAL, "NULL pointer passed in %s\n", __func__);
		exit(EXIT_FAILURE);
	}
#endif
	dump_file = xfopen(path, "w");
	// if matrix is on device, copy it to host
	if (in->location == DEVICE) {
		in = toHost(in);
		usingCopy = true;
	}

	// print matrix entries
	if(in->vartype == DOUBLE)
		for (i = 0; i < in->num; i++)
			fprintf(dump_file, "%f\n", in->ddata[i]);
	else if(in->vartype == COMPLEX)
		for (i = 0; i < in->num; i++)
			fprintf(dump_file, "%f + %fi\n", (float) cuCreal(in->cdata[i]), (float) cuCimag(in->cdata[i]));
	else {
		log_message(LOG_FATAL, "Unkown matrix type %d at %s:%s\n", in->vartype, __func__, __LINE__);
		exit(EXIT_FAILURE);
	}

	// if we copied to host, free our copy
	if (usingCopy) {
		delete_Matrix(in);
	}
	fclose(dump_file);
}

void print_Matrix_metadata(Matrix *m)
{
	char *str = NULL, buf[10];
	int i;
#ifdef DEBUG
	if(m == NULL) {
		log_message(LOG_FATAL, "NULL pointer passed in %s\n", __func__);
		exit(EXIT_FAILURE);
	}
#endif
	str = const_append("UNKNOWN TYPE ", itoa(m->vartype, buf));

	printf("m->vartype: %s\n", m->vartype == DOUBLE ? "DOUBLE" : m->vartype == COMPLEX ? "COMPLEX" : str);
	free(str);
	str = const_append("UNKNOWN LOCATION ", itoa(m->location, buf));

	printf("m->location: %s\n", m->location == DEVICE ? "DEVICE" : m->location == HOST ? "HOST" : str);
	free(str);
	printf("m->num: %zu\n", m->num);
	printf("m->size: %zu\n", m->size);
	printf("m->dims: {\n");
	for(i = 0; i < MAX_DIMS; i++)
		printf("\t%zu\n", m->dims[i]);
	puts("}");
#ifdef DEBUG
	printf("m->mat_id: %u\n", m->mat_id);
#endif
}
