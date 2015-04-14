#include <gpuNUFFT_operator_factory.hpp>

/**
 * Pass metadata to GpuNUFFTOperatorFactory
 * 
 */

extern "C" void gpuNUFFTOperatorFactory_wrapper(mat3DC ) {
	gpuNUFFT::GpuNUFFTOperatorFactory factory;
	gpuNUFFT::GpuNUFFTOperator *gpuNUFFTOp = factory.createGpuNUFFTOperator(kSpaceData,kernel_width,sector_width,osf,imgDims);
}

