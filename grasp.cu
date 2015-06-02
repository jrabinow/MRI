/*
 * C/CUDA implementation of GRASP.
 * Emma ????, Felix Moody, & Julien Rabinow
 * Fall 2014-Spring 2015
 */

/* System headers */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h> // for square root
#include <time.h> // for benchmarking
#include <complex.h> // for complex double type and operations
/* CUDA headers */
#include <cuda_runtime.h>
#include "cublas_v2.h" // CUBLAS
#include <cuComplex.h> // for cuDoubleComplex type and operations
/* Our headers */
#include "matrix.h" // host and device matrix metadata types
#include "cudaErr.h" // cuda and cublas error handlers
#include "TVTemp.h" // total variate temporal operator
#include "multicoilGpuNUFFT.cpp" // multicoil nonuniform FFT operator

/* CORRECT LINKING DEMO */
//#include <cuda_utils.hpp>
  

typedef struct {
	matrixC * read; // k space readings
	int num_spokes; // spokes per frame
	int num_frames; // frames in data
	double lambda; // trade off control TODO: between what?
	double l1Smooth; // TODO: find out what this does
	int num_iter; // number of iterations of the reconstruction
	cublasHandle_t handle; // handle to CUBLAS context	
} param_type;


#if 0
__global__ void elementWiseMultBySqrt(cuDoubleComplex * kdata, double * w) {
    // We should only have to compute the squares of the elements of w
    // one time and use the result for all slices of kdata
    int i = threadIdx.x * blockIdx.x;
    int j = blockIdx.y;
    cuDoubleComplex sqrtofelement = make_cuDoubleComplex(sqrt(w[i]), 0);
    // possible overflow error with cuCmul (see cuComplex.h)
    kdata[j] = cuCmul(kdata[j], sqrtofelement); // WARNING
}

#endif

/*
 * l1norm
 */

void l1norm(matrixC * traj,
		matrixC * sens,
		matrixC * read,
		matrix * comp,
		param_type * param) {

	print_matrixC(traj, 0, 20);
	print_matrixC(sens, 0, 20);
	print_matrixC(read, 0, 20);
	print_matrix(comp, 0, 20);

	exit(EXIT_SUCCESS);
}


#if 0


matrixC * reindex(matrixC * in, ) {
	// allocate matrix with new dimensions
	size_t new_dims[MAX_DIMS] = { };
	matrixC * out = kjlk;
	
	// loop over indices of new matrix,
	// copying old data to new matrix
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
	matrixC * read_ts = new_matrixC(read_ts_dims, host);
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
	free_matrixC(read);
	*read = *read_ts;
}

#endif

/*
 * Sort data into time series
 * It'd be cool if this was more general, but for now it's very explicit
 */

void make_time_series(matrixC * traj, matrixC * read, matrix * comp, param_type * param) {
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
	matrixC * read_ts = new_matrixC(read_ts_dims, host);

	// loop over indices of read_ts
	size_t * read_ts_coord;
	size_t read_coord[MAX_DIMS];
	size_t read_idx;
	for (size_t i = 0; i < read_ts->num; i++) {
		// convert read_ts index to read_ts coordinate
		read_ts_coord = I2C(i, read_ts->dims);
		// convert read_ts coordinate to read coordinate
		read_coord[0] = read_ts_coord[0];
		read_coord[1] = read_ts_coord[1] + read_ts_coord[3]*read_ts->dims[1];
		read_coord[2] = read_ts_coord[2];
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
	matrixC * read_copy = read;
	*read = *read_ts;
	free_matrixC(read_copy);
}



/*
 * Normalize a complex matrix on host in place so maximum modulus is 1
 * This is "normalize" as in peak normalization in audio
 * and video processing, not like in linear algebra. In the
 * former we uniformly scale entries so that the greatest
 * entry equals some value (in this case 1). In the latter
 * we uniformly scale entries so that *the norm* of the
 * matrix--treated as a vector--equals 1 (and always 1)
 * 
 */

void normalize(matrixC * mat) {
	// Make sure we get a host matrix
	if (mat->location == device) {
		printf("Error: normalize only for host matrices\n");
	}

	// Find maximum modulus
	double max_mod = 0;
	for (size_t i = 0; i < mat->num; i++) {
		if (cuCabs(mat->data[i]) > max_mod) {
			max_mod = cuCabs(mat->data[i]);
		}
	}

	// Scale entries by maximum modulus
	for (size_t i = 0; i < mat->num; i++) {
		mat->data[i] = cuCdiv(mat->data[i], make_cuDoubleComplex(max_mod, 0.0));
	}

	/*

	This is (probably nonworking) parallel version, but since it's only run once
	it's probably not worth it to transfer to device and back

	// sens=sens/max({|x|: x entry in sens})
	// scale entries of sens so that maximum modulus = 1
	size_t max_mod_idx;
	cuDoubleComplex max_mod_num;
	cublasErrChk(cublasIzamax(param->handle, mat->num, mat->data, 1, &max_mod_idx));
	cudaErrChk(cudaMemcpy(&max_mod_num,
			&(mat->data[max_mod_idx]),
			mat->size, cudaMemcpyDeviceToHost));
	const double inv_max_mod = 1/cuCabs(max_mod_num);
	cublasErrChk(cublasZdscal(handle, b1.t, &inv_max_mod, b1.d, 1));

	*/
}




/*
* Load data from file and save to array structs
* (in this case the data is from a matlab .mat file read by a custom
* script "convertmat.c" using the matio library
* and then directly written to files by fwrite)
*/

void load_data(matrixC ** traj,
		matrixC ** sens,
		matrixC ** read,
		matrix ** comp,
		param_type * param) {

	// Input data size (based on the k space readings matrix)
	// 1st dim: number of samples per reading per coil
	// 2nd dim: number of readings
	// 3rd dim: number of coils
	// TODO: should these be constants?
	size_t dims[MAX_DIMS] = {768, 600, 12};

	// Make auxillary matrix data sizes
	size_t dims_sens[MAX_DIMS] = {dims[0] / 2, dims[0] / 2, dims[2]};
	size_t dims_no_coils[MAX_DIMS] = {dims[0], dims[1]};

	// allocate matrices on host
	*traj = new_matrixC(dims_no_coils, host);
	*sens = new_matrixC(dims_sens, host);
	*read = new_matrixC(dims, host);
	*comp = new_matrix(dims_no_coils, host);

	// open matrix files
	// these were pulled from liver_data.mat by matio and convertmat
	//FILE * meta_file = fopen("./liver_data/metadata", "rb");
	FILE * traj_file = fopen("./liver_data/k.matrix", "rb");
	FILE * sens_file = fopen("./liver_data/b1.matrix", "rb");
	FILE * read_file = fopen("./liver_data/kdata.matrix", "rb");
	FILE * comp_file = fopen("./liver_data/w.matrix", "rb");

	// load matrices onto host
	fread((*traj)->data, (*traj)->size, (*traj)->num, traj_file);
	fread((*sens)->data, (*sens)->size, (*sens)->num, sens_file);
	fread((*read)->data, (*read)->size, (*read)->num, read_file);
	fread((*comp)->data, (*comp)->size, (*comp)->num, comp_file);

	// copy matrices to device
	toDeviceC(*traj);
	toDeviceC(*sens);
	toDeviceC(*read);
	toDevice(*comp);	
}

/* Just preprocessing, no GPU
 * All the good stuff is in CSL1Nlg()
 */

int main(int argc, char **argv) {

	// Input data metadata structs (defined in matrix.h)
	matrixC * traj; // trajectories through k space (k)
	matrixC * sens; // coil sensitivities (b1)
	matrixC * read; // k space readings (kdata)
	matrix * comp; // density compensation (w)

	// Reconstruction parameters
	param_type * param = (param_type *)malloc(sizeof(param_type));
	param->num_spokes = 21; // spokes (i.e. readings) per frame (Fibonacci number)
	param->num_iter = 8; // number of iterations of the reconstruction
	cublasErrChk(cublasCreate(&(param->handle))); // create cuBLAS context
	
	// Load data
	load_data(&traj, &sens, &read, &comp, param);

	// Emma's l1norm testing function
	if (false) {
		l1norm(traj, sens, read, comp, param);
	}
	
	// normalize coil sensitivities
	normalize(sens);

	// multiply readings by density compensation
	// TODO (Julien): write this function

	// crop data so that spokes divide evenly into frames with none left over
	param->num_frames = read->dims[1] / param->num_spokes;
	size_t new_dims_read[MAX_DIMS] = {read->dims[0],
			param->num_frames*param->num_spokes,
			read->dims[2] };
	size_t new_dims_no_coils[MAX_DIMS] = { new_dims_read[0],
			new_dims_read[1] };

	traj = crop_matrixC(traj, new_dims_no_coils);
	read = crop_matrixC(read, new_dims_read);
	comp = crop_matrix(comp, new_dims_no_coils);

	// sort into time series
	// TODO (Julien): get this working
	//make_time_series(traj, read, comp, param);

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
	print_matrixC(traj, 0, 20);
	printf("\n----Coil sensitivities aka sens aka b1----\n");
	print_matrixC(sens, 0, 20);
	printf("\n----K-space readings aka read aka kdata----\n");
	print_matrixC(read, 0, 20);
	printf("\n----Density compensation aka comp aka w----\n");
	print_matrix(comp, 0, 20);
	
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
	// recon_cs = CSL1NlCg(recon_cs,param);
	// }
	// long elapsedtime = (clock_gettime(CLOCK_MONOTONIC, tv_nsec) - starttime)/1000000;

	// recon_nufft=flipdim(recon_nufft,1);
	// recon_cs=flipdim(recon_cs,1);


	// destroy cuBLAS context
	cublasDestroy(param->handle);

	// free memory
	free_matrixC(traj);
	free_matrixC(sens);
	free_matrixC(read);
	free_matrix(comp);

	return 0;
}
