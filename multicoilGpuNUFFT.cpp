/*
 * Multicoil wrapper for gpuNUFFT
 * 
 * In the matlab version of grasp which uses Jeff Fessler's nufft_toolbox,
 * each coil in each frame is transformed as a unit,
 * i.e., num_spokes columns in one coil are treated as one vector.
 * So, we precompute one nufft for each frame, with our trajectory being
 * the concatenation of the trajectories for each spoke in that frame.
 * Then, when the reverse nufft is called we run the precomputed nufft
 * for each coil in each frame, combining the coils in each frame.
 *
 * gpuNUFFT is setup similar to Fessler's nufft_toolbox: we have a
 * gpuNUFFTOperatorFactory class that does the precomputation step,
 * producing an operator for some trajectory,
 * then we can run that operator as many times as we want on data from that trajectory.
 * TODO:
 * How to initialize gpuNUFFTOperatorFactory? Have options:
 * 	useTextures: flag to indicate texture interpolation
 * 	useGpu: Flag to indicate gpu usage for precomputation
 * 	balaceWorkload: flag to indicate load balancing
 * For now, using default of all options true
 * Which gpuNUFFTOperatorFactory method to use to make operator?
 * 	createGpuNUFFTOperator
 * 	loadPrecomputeGpuNUFFTOperator: why can't we just save previous calls
 * 		in memory?
 * 	
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "matrix.h"
#include "cudaErr.h"
#include "gpuNUFFT_operator_factory.hpp"

// A global multidimensional array of pointers to gpuNUFFT operators
gpuNUFFT::GpuNUFFTOperator *** gpuNUFFTOps;

extern void createMulticoilGpuNUFFTOperator(matrixC * traj,
		matrix * comp,
		matrixC * sens,
		int kernel_width,
		int sector_width,
		double oversample_ratio) {
	
	// Create operator factory with default constructor:
	// textures false, balanced false, preprocessing on gpu false
	// TODO: find optimal settings. Although, this is just for preprocessing
	// so not very important (grasp measures runtime of reconstruction not
	// counting preprocessing)
	gpuNUFFT::GpuNUFFTOperatorFactory factory;
	
	// Create array containers to store data for each frame
	gpuNUFFT::Array<DType> kSpaceTraj;
	gpuNUFFT::Array<DType> densCompData;
	gpuNUFFT::Array<DType2> sensData;
	
	// Allocate array to store operator pointers
	for (size_t i = 0; i < traj->dims[2]; i++) {
		gpuNUFFTOps[i] = (gpuNUFFT::GpuNUFFTOperator **)malloc(
				sens->dims[2]*sizeof(gpuNUFFT::GpuNUFFTOperator *));
	}

	// Allocate temporary space for data
	// This is under the assumption that createGpuNUFFTOperator()
	// makes a copy of it's input parameters,
	// so it's okay to change them after each iteration of the
	// below for loop
	kSpaceTraj.data = (DType *)malloc(2*traj->num*sizeof(DType));
	densCompData.data = (DType *)malloc(comp->num*sizeof(DType));
	sensData.data = (DType2 *)malloc(sens->num*sizeof(DType2));
	
	// Cast some arguments
	const IndType kernelWidth = (IndType)kernel_width;
	const IndType sectorWidth = (IndType)sector_width;
	const DType osf = (DType)oversample_ratio;

	// Pull image dimensions from coil sensitivity maps,
	// casting from size_t to IndType, which is itself
	// just a typedef of unsigned int
	// TODO: ensure there's no overflow here
	gpuNUFFT::Dimensions imgDims;
	imgDims.width = (IndType)sens->dims[0];
	imgDims.height = (IndType)sens->dims[1];
	
	/* 
	 * Create operator for each coil in each frame, treating the whole
	 * frame as one vector.
	 * This is different from Fessler's nufft_toolbox where the operator
	 * was made without specifying the coil and so only one operator
	 * needed for each frame, resused for each coil 
	 */		

	size_t traj_coord[MAX_DIMS] = { 0, 0, 0 };
	size_t comp_coord[MAX_DIMS] = { 0, 0, 0 };
	size_t sens_coord[MAX_DIMS] = { 0, 0, 0 };
	size_t traj_index;
	size_t comp_index;
	size_t sens_index;
	// for each frame
	for (size_t i = 0; i < traj->dims[2]; i++) {
		// set coordinate to start of frame
		traj_coord[2] = i;
		comp_coord[2] = i;
		
		// Copy and cast trajectories: traj from grasp is an array of
		// cuDoubleComplex numbers. gpuNUFFT expects interleaved data:
		// (real(traj[0]), imag(traj[0]), real(traj[1]), imag(traj[1]), etc)
		// where each entry is of type DType (typedef of float).
		// We may be able to relate the
		// underlying data directly, but for
		// now we allocate new array and explicitly cast each element
		// Also might need to rescale traj first, since in example
		// it says gpuNUFFT expects values between -.5 and .5, whereas I
		// think grasp uses values between 0 and 1
		for (size_t j = 0; j < traj->dims[0]*traj->dims[1]; j++) {
			traj_index = C2I(traj_coord, traj->dims) + j; 
			kSpaceTraj.data[2*j] = (DType)cuCreal(traj->data[traj_index]);
			kSpaceTraj.data[2*j + 1] = (DType)cuCimag(traj->data[traj_index]);
		}

		// Cast density compensation: comp from grasp is an array
		// of doubles. gpuNUFFT expects an array of DType, which is
		// a typedef of float.
		// TODO: every column of comp is the same, so it could
		// be that gpuNUFFT expects just one column
		// For now, we assume that it's possible to have
		// different columns
		// Also we should be taking the square root of this
		for (size_t j = 0; j < comp->dims[0]*comp->dims[1]; j++) {
			comp_index = C2I(comp_coord, comp->dims) + j;
			densCompData.data[j] = (DType)comp->data[comp_index];
		}

		// for each coil
		for (size_t j = 0; j < comp->dims[2]; j++) {
			// set coordinate to start of the coil
			sens_coord[2] = j;

			// Cast coil sensitivities: sens from grasp is an array
			// of cuDoubleComplex numbers. gpuNUFFT expects an array of
			// DType2, which is a typedef of cuda's float2, which (I think)
			// is a struct with two floats x and y
			for (size_t k = 0; k < sens->dims[0]*sens->dims[1]; k++) {
				sens_index = C2I(sens_coord, sens->dims) + k;
				sensData.data[k].x = (float)cuCreal(sens->data[sens_index]);
				sensData.data[k].y = (float)cuCimag(sens->data[sens_index]);
			}
			
			// create operator for this frame and this coil
			gpuNUFFTOps[i][j] = factory.createGpuNUFFTOperator(kSpaceTraj,
					densCompData,
					sensData,
					kernelWidth,
					sectorWidth,
					osf,
					imgDims);
		}
		// TODO: free temp data
	}
}
		





	/*  multicoil wrapper for createGpuNUFFTOperator()
 	 */
#if 0
	void createMulticoilGpuNUFFTOperator(matrixC * traj,
			matrix * comp,
			matrixC * sens,
			int kernel_width,
			int sector_width,
			double oversample_ratio) {
				
		// Create operator factory with default constructor:
		// textures, balanced, and gpu true
		gpuNUFFT::GpuNUFFTOperatorFactory factory;
		
		// Create array containers to store data for each frame
		gpuNUFFT::Array<DType> kSpaceTraj;
		gpuNUFFT::Array<DType> densCompData;
		gpuNUFFT::Array<DType> sensData;
		
		// Allocate temporary space for data
		// This is under the assumption that createGpuNUFFTOperator()
		// makes a copy of it's input parameters,
		// so it's okay to change them after each iteration of the
		// below for loop
		kSpaceTraj.data = (DType *)malloc(2*traj->num*sizeof(DType));
		densCompData.data = (DType *)malloc(comp->num*sizeof(DType));
		sensData.data = (DType2 *)malloc(sens->num*sizeof(DType2));
		
		// Cast arguments
		const IndType kernelWidth = (IndType)kernel_width;
		const IndType sectorWidth = (IndType)sector_width;
		const DType osf = (DType)oversample_ratio;

		// Pull image dimensions from coil sensitivity maps,
		// casting from size_t to IndType, which is itself
		// just a typedef of unsigned int
		gpuNUFFT::Dimensions imgDims;
		imgDims.width = (IndType)sens->dims[0];
		imgDims.height = (IndType)sens->dims[1];
		
		/* 
 		 * Create operator for each frame, treating the whole
 		 * frame as one image, concatenating trajectories for each spoke
		 */		

		for (size_t i = 0; i < traj->dims[3]; i++) {

			// Copy and cast trajectories: traj from grasp is an array of
			// cuDoubleComplex numbers. gpuNUFFT expects interleaved data:
			// (real(traj[0]), imag(traj[0]), real(traj[1]), imag(traj[1]), etc)
			// where each entry is of type DType (typedef of float).
			// We may be able to relate the
			// underlying data directly, but for
			// now we allocate new array and explicitly cast each element
			// Also might need to rescale traj first, since in example
			// it says gpuNUFFT expects values between -.5 and .5, whereas I
			// think grasp uses values between 0 and 1
			for (size_t i = ; i < traj->num; i++) {
				kSpaceTraj.data[2*i] = (DType)cuCreal(traj->data[i]);
				kSpaceTraj.data[2*i + 1] = (DType)cuCimag(traj->data[i+1]);
			}

			// Cast density compensation: comp from grasp is an array
			// of doubles. gpuNUFFT expects an array of DType, which is
			// a typedef of float.
			for (size_t i = 0; i < comp->num; i++) {
				densCompData.data[i] = (DType)comp->data[i];
			}

			// Cast coil sensitivities: sens from grasp is an array
			// of cuDoubleComplex numbers. gpuNUFFT expects an array of
			// DType2, which is a typedef of cuda's float2, which (I think)
			// is a struct with two floats x and y
			for (size_t i = 0; i < sens->num; i++) {
				sensData.data[i].x = (float)cuCreal(sens->data[i]);
				sensData.data[i].y = (float)cuCimag(sens->data[i]);
			}

			


#endif


/*
  int kernel_width = 3;
  float osf = 1.25;//oversampling ratio
  int sector_width = 8;
  
  //Data
  int data_entries = 2;
  DType2* data = (DType2*) calloc(data_entries,sizeof(DType2)); //2* re + im
  data[0].x = 5;//Re
  data[0].y = 0;//Im
  data[1].x = 1;//Re
  data[1].y = 0;//Im
  //Coords
  //Scaled between -0.5 and 0.5
  //in triplets (x,y,z) as structure of array
  //p0 = (0,0,0)
  //p1 0 (0.25,0.25,0.25)
  DType* coords = (DType*) calloc(3*data_entries,sizeof(DType));//3* x,y,z
  coords[0] = 0.00; //x0
  coords[1] = 0.25; //x1
  
  coords[2] = 0.00; //y0
  coords[3] = 0.25; //y0
  
  coords[4] = 0.00; //z0
  coords[5] = 0.25; //z1
  //Input data array, complex values
  gpuNUFFT::Array<DType2> dataArray;
  dataArray.data = data;
  dataArray.dim.length = data_entries;
  
  //Input array containing trajectory in k-space
  gpuNUFFT::Array<DType> kSpaceData;
  kSpaceData.data = coords;
  kSpaceData.dim.length = data_entries;
  gpuNUFFT::Dimensions imgDims;
  imgDims.width = 64;
  imgDims.height = 64;
  imgDims.depth = 64;
  //precomputation performed by factory
  gpuNUFFT::GpuNUFFTOperatorFactory factory; 
  gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceData,kernel_width,sector_width,osf,imgDims);
  //Output Array
  gpuNUFFT::Array<CufftType> imgArray;
  
  //Perform FT^H Operation
  imgArray = gpuNUFFTOp->performGpuNUFFTAdj(dataArray);
  
  //Output Image
  CufftType* gdata = imgArray.data;
  
  //Perform FT Operation
  gpuNUFFT::Array<CufftType> kSpace = gpuNUFFTOp->performForwardGpuNUFFT(imgArray);
  
  printf("contrast %f \n",kSpace.data[0].x/kSpace.data[1].x);
  free(data);
  free(coords);
  free(gdata);
  delete gpuNUFFTOp;

*/
