/*
// in is a 3 dimensional array on device such that one slice fits in a grid
// and there are fewer slices than threads per block
// y is a 3 dim output array on device of the same size as x
// adjoint is a boolean
__global__ void TV_Temp(cuDoubleComplex * x, cuDoubleComplex * y, int adjoint) {
    if (adjoint == 1) {
        if (blockIdx.x == 1) {
            y[I3D(threadIdx.x, threadIdx.y, 1, blockDim.x, blockDim.y)]
                = -x[I3D(threadIdx.x, threadIdx.y, 1, blockDim.x, blockDim.y)];
        } else if (blockIdx.x == gridDim.x) {
            y[I3D(threadIdx.x, threadIdx.y, gridDim.x, blockDim.x, blockDim.y)]
                = x[I3D(threadIdx.x, threadIdx.y, gridDim.x-1, blockDim.x, blockDim.y)];
        } else {
            y[I3D(threadIdx.x, threadIdx.y, blockId.x, blockDim.x, blockDim.y)]
                = x[I3D(threadIdx.x, threadIdx.y, blockId.x-1, blockDim.x, blockDim.y)]
                - x[I3D(threadIdx.x, threadIdx.y, blockId.x, blockDim.x, blockDim.y)];
        }
    if (adjoint == 0) {
            y[I3D(threadIdx.x, threadIdx.y, blockId.x, blockDim.x, blockDim.y)]
                = x[I3D(threadIdx.x, threadIdx.y, blockId.x+1, blockDim.x, blockDim.y)]
                - x[I3D(threadIdx.x, threadIdx.y, blockId.x, blockDim.x, blockDim.y)];
    }
}
*/