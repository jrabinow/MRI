/*
 * Multicoil wrapper for gpuNUFFT
 * Allows the calling of C++ function from C and converts datatypes
 * 
 */


#include <gpuNUFFT_operator_factory.hpp>

gpuNUFFTOperatorFactory



/*
/* 
/*
extern "C" void gpuNUFFTOperatorFactory_wrapper() {
	gpuNUFFT::GpuNUFFTOperatorFactory factory;
}

	gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceData,kernel_width,sector_width,osf,imgDims);


gpuNUFFT::GpuNUFFTOperatorFactory factory; 
gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceData,kernel_width,sector_width,osf,imgDims);
gpuNUFFTOp->performGpuNUFFTAdj(dataArray)
gpuNUFFTOp->performForwardGpuNUFFT(imgArray)



/* Questions about using gpuNUFFT
 * --Should 
/* Relavent gpuNUFFT datatypes:


*/


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


%	om [M,d]	"digital" frequencies in radians
%	Nd [d]		image dimensions (N1,N2,...,Nd)
%	Jd [d]		# of neighbors used (in each direction)
%	Kd [d]		FFT sizes (should be >= N1,N2,...)

%	n_shift [d]	n = 0-n_shift to N-1-n_shift (must be first)



kSpaceTraj	coordinate array of sample locations
kernelWidth	interpolation kernel size in grid units
sectorWidth	sector width
osf	grid oversampling ratio
imgDims	image dimensions (problem size) 

*/