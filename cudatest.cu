#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

__global__ void VecAdd(int* A) {
    int i = threadIdx.x;
    //C[i] = A[i] + B[i];
    A[i] = A[i] + 3;
}

int main() {
    int x[300];
    //int y[300];
    //int z[300];
    int* x_d;
    //int* y_d;
    //int* z_d;

    cudaMalloc(&x_d, 300*sizeof(int));
    //cudaMalloc(&y_d, 300*sizeof(int));
    //cudaMalloc(&z_d, 300*sizeof(int));

    int i;
    for(i=0; i<300; i++) {
        x[i] = 100;
        //x[i] = random(1000);
        //y[i] = random(1000);
    }

    cudaMemcpy(x, x_d, sizeof(x), cudaMemcpyHostToDevice);

    VecAdd<<<1, 300>>>(x_d);

    cudaMemcpy(x_d, x, sizeof(x), cudaMemcpyDeviceToHost);

    int j;
    for(j=0; j<300; j++) {
        printf("%d ", x[j]);
    }
}
