/*
 * C/CUDA implementation of GRASP.
 * Emma ????, Felix Moody, & Julien Rabinow
 * Fall 2014-Spring 2015
 */

#include <grasp.h>

/*
 * Sort data into time series
 * It'd be cool if this was more general, but for now it's very explicit
 */
Matrix* make_time_series(Matrix* traj, Matrix* read, Matrix* comp, Param_type* param)
{
	Coordinate read_ts_coord;
	size_t read_coord[MAX_DIMS];
	size_t read_idx;
	// Since for traj and comp we're just splitting the last dimensions
	// we get away with just reindexing the same underlying data,
	traj->dims[1] = comp->dims[1] = param->num_spokes;
	traj->dims[2] = comp->dims[2] = param->num_frames;
	// But read requires reordering the data
	// allocate new matrix read_ts
	size_t read_ts_dims[MAX_DIMS] = {
		read->dims[0],
		param->num_spokes,
		read->dims[2],
		param->num_frames
	};
	Matrix* read_ts = new_Matrix(read_ts_dims, HOST, COMPLEX);

	// loop over indices of read_ts
	for (size_t i = 0; i < read_ts->num; i++) {
		// convert read_ts index to read_ts coordinate
		read_ts_coord = I2C(i, read_ts->dims);
		// convert read_ts coordinate to read coordinate
		read_coord[0] = read_ts_coord.coord[0];
		read_coord[1] = read_ts_coord.coord[1] + read_ts_coord.coord[3]*read_ts->dims[1];
		read_coord[2] = read_ts_coord.coord[2];
		// convert read coordinate to read index
		read_idx = C2I(read_coord, read->dims);
		// copy entry from read to read_ts
		read_ts[i] = read[read_idx];
		// copy column
		//memcpy(&(read_ts->data[i]),
		//		&(read->data[read_idx]),
		//		read->dims[0]*read->size);
	}
	// reassign read pointer to time series and free old data
	// we have to make a copy of the pointer to old read so we don't loose
	// track of it after the assignment
	return read_ts;
}

/*
 * Normalize a complex matrix on host in place so maximum modulus is 1
 * This is "normalize" as in peak normalization in audio
 * and video processing, not like in linear algebra. In the
 * former we uniformly scale entries so that the greatest
 * entry equals some value (in this case 1). In the latter
 * we uniformly scale entries so that *the norm* of the
 * matrix--treated as a vector--equals 1 (and always 1)
 */
void normalize(Matrix* mat)
{
	double max_mod_double = 0;
	cuDoubleComplex max_mod;

	// Make sure we get a host matrix of complex values
	if (mat->location == DEVICE) {
		log_message(LOG_FATAL, "normalize only for host matrices");
		exit(EXIT_FAILURE);
	} else if(mat->vartype != COMPLEX) {
		log_message(LOG_FATAL, "matrix of non-complex values passed");
		exit(EXIT_FAILURE);
	}
	// Find maximum modulus
	for (size_t i = 0; i < mat->num; i++) {
		if (cuCabs(mat->cdata[i]) > max_mod_double) {
			max_mod_double = cuCabs(mat->cdata[i]);
		}
	}
	max_mod = make_cuDoubleComplex(max_mod_double, (double) 0);
	// Scale entries by maximum modulus
	for (size_t i = 0; i < mat->num; i++)
		mat->cdata[i] = cuCdiv(mat->cdata[i], max_mod);

	/*
	Nonworking parallel version. Since this code is only run a single time,
	it's probably not worth transferring to and from device

	// sens=sens/max({|x|: x entry in sens})
	// scale entries of sens so that maximum modulus = 1
	size_t max_mod_idx;
	cuDoubleComplex max_mod_num;
	cublasErrChk(cublasIzamax(param->handle, mat->num, mat->data, 1,
		&max_mod_idx));
	cudaErrChk(cudaMemcpy(&max_mod_num, &(mat->data[max_mod_idx]),
			mat->size, cudaMemcpyDeviceToHost));
	const double inv_max_mod = 1/cuCabs(max_mod_num);
	cublasErrChk(cublasZdscal(handle, b1.t, &inv_max_mod, b1.d, 1));
	*/
}

void apply_density_compensation(Matrix *read, Matrix *comp)
{
	size_t j = 0;
	cuDoubleComplex *sqrt_comp = NULL;

	if(read == NULL || comp == NULL) {
		fprintf(stderr, "Error: NULL pointer passed in %s\n", __func__);
		exit(EXIT_FAILURE);
	}
	sqrt_comp = (cuDoubleComplex*) xmalloc(comp->num * sizeof(cuDoubleComplex));

	for(size_t i = 0; i < comp->num; i++)
		sqrt_comp[i] = make_cuDoubleComplex(sqrt(comp->ddata[i]), (double) 0);
	for(size_t i = 0; i < read->num; i += j)
		for(j = 0; j < comp->num; j++)
			read->cdata[i + j] = cuCmul(read->cdata[i + j], sqrt_comp[j]);
	free(sqrt_comp);
}

Matrix *load_matrix_from_file(const char *path, size_t *dims, varFlag vartype)
{
	FILE *input = NULL;
	Matrix *mat = new_Matrix(dims, HOST, vartype);

	input = xfopen(path, "rb");
	if(fread(mat->data, mat->size, mat->num, input) != mat->num) {
		failwith("Error: failed to read entire matrix from file");
		exit(EXIT_FAILURE);
	}
	fclose(input);

	return mat;
}

/*
 * Load data from file and save to array structs
 * (in this case the data is from a matlab .mat file read by a custom
 * script "convertmat.c" using the matio library
 * and then directly written to files by fwrite)
 */
void load_data(Matrix ** traj, Matrix ** sens, Matrix ** read, Matrix ** comp,
		Param_type * param)
{
	// Input data size (based on the k space readings matrix)
	// 1st dim: number of samples per reading per coil
	// 2nd dim: number of readings
	// 3rd dim: number of coils
	// TODO: should these be constants?
	/* NO they most definitely should not if we can avoid it. How could we
	 * determine matrix dimensions, are they saved in the file along with
	 * other metadata by any chance ? */
	size_t dims[MAX_DIMS] = { 768, 600, 12 };

	// Make auxillary matrix data sizes
	size_t dims_sens[MAX_DIMS] = {
		dims[0] / 2,
		dims[0] / 2,
		dims[2]
	};
	size_t dims_no_coils[MAX_DIMS] = { dims[0], dims[1] };

	*traj = load_matrix_from_file(TRAJ_FILE_PATH, dims_no_coils, COMPLEX);
	*sens = load_matrix_from_file(SENS_FILE_PATH, dims_sens, COMPLEX);
	*read = load_matrix_from_file(READ_FILE_PATH, dims, COMPLEX);
	*comp = load_matrix_from_file(COMP_FILE_PATH, dims_no_coils, DOUBLE);
}

/*
 * Just preprocessing, no GPU
 * All the good stuff is in CSL1Nlg()
 */
int main(int argc, char **argv)
{
	// Input data metadata structs (defined in matrix.h)
	Matrix* traj = NULL;	// trajectories through k space (k)
	Matrix* sens = NULL;	// coil sensitivities (b1)
	Matrix* read = NULL;	// k space readings (kdata)
	Matrix* read_ts = NULL;	// read sorted into time series
	Matrix* comp = NULL; 	// density compensation (w)

	// Reconstruction parameters
	Param_type param;
	param.num_spokes = NUM_SPOKES;		// spokes (i.e. readings) per frame (Fibonacci number)
	param.num_iter = NUM_ITERATIONS;	// number of iterations of the reconstruction
	cublasErrChk(cublasCreate(&param.handle)); // create cuBLAS context
#ifdef DEBUG
	init_log(stderr, LOG_DEBUG);
#else
	init_log(stderr, LOG_FATAL);
#endif
	load_data(&traj, &sens, &read, &comp, &param);
	normalize(sens);
	apply_density_compensation(read, comp);

	// crop data so that spokes divide evenly into frames with none left over
	param.num_frames = read->dims[1] / param.num_spokes;
	size_t new_dims_read[MAX_DIMS] = {
		read->dims[0],
		param.num_frames * param.num_spokes,
		read->dims[2]
	};
	size_t new_dims_no_coils[MAX_DIMS] = {
		new_dims_read[0],
		new_dims_read[1]
	};

	traj = crop_Matrix(traj, new_dims_no_coils);
	read = crop_Matrix(read, new_dims_read);
	comp = crop_Matrix(comp, new_dims_no_coils);

	// sort into time series
	// TODO (Julien): get this working
/*	read_ts = make_time_series(traj, read, comp, &param);
	delete_Matrix(read); */

	// gpuNUFFT operator
	if (false) {
		int kernel_width = 3;
		int sector_width = 8;
		double oversample_ratio = 1.25;

		createMulticoilGpuNUFFTOperator(traj,
				comp,
				sens,
				kernel_width,
				sector_width,
				oversample_ratio);
	}

	// print matrices
	printf("\n----K-space trajectories aka traj aka k----\n");
	print_Matrix(traj, 0, 20);
	printf("\n----Coil sensitivities aka sens aka b1----\n");
	print_Matrix(sens, 0, 20);
	printf("\n----K-space readings aka read aka kdata----\n");
//	print_Matrix(read_ts, 0, 20);
	print_Matrix(read, 0, 20);
	printf("\n----Density compensation aka comp aka w----\n");
	print_Matrix(comp, 0, 20);

	// WORKING UP TO HERE

	/* CORRECT LINKING DEMO */
	//bindTo1DTexture("symbol", NULL, 0);

	// GPU block and grid dimensions
	/*
	int bt = 512; // max threads per block total
	int bx = 512; // max threads per block x direction
	int by = 512; // max threads per block y direction
	int bz = 64; // max threads per block z direction
	int gx = 65535;
	int gy = 65535;
	int gz = 65535;
	*/

	/*
	// for ch=1:nc,kdata(:,:,ch)=kdata(:,:,ch).*sqrt(w);endc
	// i.e. multiply each of the 12 slices of kdata element-wise by sqrt(w)
	dim3 numBlocks((kdata.x*kdata.y)/bt, kdata.z);
	elementWiseMultBySqrt<<<numBlocks, bt>>>(kdata.d, w.d);
	*/

	/*
	// %%%%% multicoil NUFFT operator
	param.E=MCNUFFT(ku,wu,b1);
	*/

	/*
	// %%%%% undersampled data
	param.y=kdatau;
	// clear kdata kdatau k ku wu w
	*/

	/*
	// %%%%% nufft recon
	// ' := conjugate transpose; * := matrix multiplication
	// ' and * are overloaded, defined in @MCNUFFT
	// what's the order of operations, or does it matter?
	mat3DC recon_nufft=param.E'*param.y;
	*/

	/*
	//param.lambda = 0.25*max(abs(recon_nufft(:)));
	*/

	// fprintf('\n GRASP reconstruction \n')

	// long starttime = clock_gettime(CLOCK_MONOTONIC, tv_nsec);
	// mat3DC recon_cs=recon_nufft;
	// for (i = 0; i < 4; i++) {
	// recon_cs = CSL1NlCg(recon_cs,&param);
	// }
	// long elapsedtime = (clock_gettime(CLOCK_MONOTONIC, tv_nsec) - starttime)/1000000;

	// recon_nufft=flipdim(recon_nufft,1);
	// recon_cs=flipdim(recon_cs,1);


	// destroy cuBLAS context
	cublasDestroy(param.handle);

	// free memory
	delete_Matrix(traj);
	delete_Matrix(sens);
//	delete_Matrix(read_ts);
	delete_Matrix(read);
	delete_Matrix(comp);
	cudaDeviceReset();

	return 0;
}

#if 0
__global__ void elementWiseMultBySqrt(cuDoubleComplex * kdata, double * w) {
    // We should only have to compute the squares of the elements of w
    // one time and use the result for all slices of kdata
    int i = threadIdx.x * blockIdx.x;
    int j = blockIdx.y;
    cuDoubleComplex sqrtofelement = make_cuDoubleComplex(sqrt(w[i]), (double) 0);
    // possible overflow error with cuCmul (see cuComplex.h)
    kdata[j] = cuCmul(kdata[j], sqrtofelement); // WARNING
}

MatrixC * reindex(Matrix * in, ) {
	// allocate Matrix with new dimensions
	size_t new_dims[MAX_DIMS] = { };
	Matrix * out = kjlk;

	// loop over indices of new Matrix,
	// copying old data to new Matrix
	size_t * new_coord;
	size_t old_coord[MAX_DIMS];
	for (size_t i = 0; i < out->num; i++) {
		// convert new index to new coordinate
		new_coord = I2C(i, new->dims);
		// convert new coordinate to old coordinate
		old_coord



	// Since for traj and comp we're just splitting the last dimensions
	// we get away with just reindexing the same underlying data,
	traj->dims[1] = param->num_spokes;
	traj->dims[2] = param->num_frames;
	comp->dims[1] = param->num_spokes;
	comp->dims[2] = param->num_frames;

	// But read requires reordering the data
	// allocate new matrix read_ts
	size_t read_ts_dims[MAX_DIMS] = { read->dims[0],
			param->num_spokes,
			read->dims[2],
			param->num_frames };
	Matrix * read_ts = new_Matrix(read_ts_dims, host);
	// loop over the first entries of the columns of read_ts
	size_t * read_ts_coord;
	size_t read_coord[MAX_DIMS];
	size_t read_idx;
	for (size_t i = 0; i < read_ts->num; i += read_ts->dims[0]) {
		// convert read_ts index to read_ts coordinate
		read_ts_coord = I2C(i, read_ts->dims);
		// find equivalent read coordinate
		read_coord[0] = read_ts_coord[0];
		read_coord[1] = read_ts_coord[1] + read_ts_coord[3]*read_ts->dims[1];
		read_coord[2] = read_ts_coord[2];
		// convert read coordinate to read index
		read_idx = C2I(read_coord, read->dims);
		// copy column
		memcpy(&(read_ts->data[i]),
				&(read->data[read_idx]),
				read->dims[0]*read->size);
	}
	// reassign read pointer to time series and free old data
	delete_Matrix(read);
	*read = *read_ts;
}

#endif

