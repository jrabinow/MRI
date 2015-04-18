/*
 * C/CUDA implementation of GRASP.
 * Emma ????, Felix Moody, & Julien Rabinow
 * Fall 2014-Spring 2015
 */

/* System headers */
#include <stdlib.h>
#include <stdio.h>
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
#include "multicoilGpuNUFFT.hpp" // multicoil nonuniform FFT operator

/* CORRECT LINKING DEMO */
#include <cuda_utils.hpp>
  
/*
typedef struct {
    cuDoubleComplex * y; // kdatau
    double lambda; // trade off control TODO: between what?
    double l1Smooth; // TODO: find out what this does
    int nite = 8; // TODO: find out what this does
} param_t;
*/

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

int main(int argc,char **argv) {
    // Data size (determines gpu optimization, so don't change lightly!)
    int nx = 768;
    int ntviews = 600;
    int nc = 12;

#if 0

    /* CORRECT LINKING DEMO */
    bindTo1DTexture("symbol", NULL, 0);

    // GPU block and grid dimensions
    int bt = 512; // max threads per block total
    int bx = 512; // max threads per block x direction
    int by = 512; // max threads per block y direction
    int bz = 64; // max threads per block z direction
    int gx = 65535;
    int gy = 65535;
    int gz = 65535;
    
    int i, j, l, m; // general loop indices (skipped k)

    cublasHandle_t handle; // handle to CUBLAS context

    cudaErrChk(cudaSetDevice(0));

    //  number of spokes to be used per frame (Fibonacci number)
    int nspokes = 21;

    // %%%%%% load radial data
    // open matrix files
    // these were pulled from liver_data.mat by convertmat
    FILE * meta_file = fopen("./liver_data/metadata", "rb");
    FILE * b1_file = fopen("./liver_data/b1.matrix", "rb");
    FILE * k_file = fopen("./liver_data/k.matrix", "rb");
    FILE * kdata_file = fopen("./liver_data/kdata.matrix", "rb");
    FILE * w_file = fopen("./liver_data/w.matrix", "rb");

    // temporarily allocate and load b1, k, kdata, and w on CPU
    cuDoubleComplex * b1_cpu = (cuDoubleComplex *)malloc((nx/2)*(nx/2)*nc * sizeof(cuDoubleComplex));
    fread(b1_cpu, sizeof(cuDoubleComplex), (nx/2)*(nx/2)*nc, b1_file);
    cuDoubleComplex * k_cpu = (cuDoubleComplex *)malloc(nx*ntviews * sizeof(cuDoubleComplex));
    fread(k_cpu, sizeof(cuDoubleComplex), nx*ntviews, k_file);
    cuDoubleComplex * kdata_cpu = (cuDoubleComplex *)malloc(nx*ntviews*nc * sizeof(cuDoubleComplex));
    fread(kdata_cpu, sizeof(cuDoubleComplex), nx*ntviews*nc, kdata_file);
    double * w_cpu = (double *)malloc(nx*ntviews * sizeof(double));
    fread(w_cpu, sizeof(double), nx*ntviews, w_file);

    // allocate b1, k, kdata, w on GPU
    mat3DC b1 = new_mat3DC(nx/2, nx/2, nc);
    mat2DC k = new_mat2DC(nx, ntviews);
    mat3DC kdata = new_mat3DC(nx, ntviews, nc);
    mat2D w = new_mat2D(nx, ntviews);
   
    // copy data from CPU to GPU
    cudaErrChk(cudaMemcpy(b1.d, b1_cpu, b1.s*b1.t, cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(k.d, k_cpu, k.s*k.t, cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(kdata.d, kdata_cpu, kdata.s*kdata.t, cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(w.d, w_cpu, w.s*w.t, cudaMemcpyHostToDevice));

    // create cuBLAS context
    cublasErrChk(cublasCreate(&handle));

    // b1=b1/max({|x|: x entry in b1})
    // scale entries of b1 so that maximum modulus = 1
    int max_mod_idx;
    cuDoubleComplex max_mod_num;
    cublasErrChk(cublasIzamax(handle, b1.t, b1.d, 1, &max_mod_idx));
    cudaErrChk(cudaMemcpy(&max_mod_num, &(b1.d[max_mod_idx]), b1.s, cudaMemcpyDeviceToHost));
    const double inv_max_mod = 1/cuCabs(max_mod_num);
    cublasErrChk(cublasZdscal(handle, b1.t, &inv_max_mod, b1.d, 1));

    // WORKING UP TO HERE

/*
    // for ch=1:nc,kdata(:,:,ch)=kdata(:,:,ch).*sqrt(w);endc
    // i.e. multiply each of the 12 slices of kdata element-wise by sqrt(w)
    dim3 numBlocks((kdata.x*kdata.y)/bt, kdata.z);
    elementWiseMultBySqrt<<<numBlocks, bt>>>(kdata.d, w.d);

    printcol_mat3DC(kdata, 0, 0); 

    // %%%%% number of frames
    int nt = ntviews/nspokes; // floor is implicit
*/
/*
    // I THINK THE FOLLOWING SECTION REPLACES THIS
    // %%%%% crop the data according to the number of spokes per frame
    // we're basically setting ntviews = nt*nspokes
    // kdata=kdata(:,1:nt*nspokes,:)
    // looping column first due to column major storage
    for (k = 0; k < nc; k++) {
        for (i = 0; i < nx; i++ {
            for (j = 0; j < nt*nspokes; j++) {
                kdata_d[I3D(i,j,k,nx,nt*nspokes)] = kdata_d[I3D(i,j,k,nx,ntviews)];
            }
         }
    }
    // k=k(:,1:nt*nspokes)
    for (i = 0; i < nx; i++ {
        for (j = 0; j < nt*nspokes; j++) {
            k_d[I2D(i,j,nt*nspokes)] = k_d[I2D(i,j,ntviews)];
        }
    }
    // w=w(:,1:nt*nspokes);
    for (i = 0; i < nx; i++ {
        for (j = 0; j < nt*nspokes; j++) {
            w_d[I2D(i,j,nt*nspokes)] = w_d[I2D(i,j,ntviews)];
        }
    }
*/
/*
    // %%%%% sort the data into a time-series
    // sort kdata, k, and w into time series kdatau, ku, and wu
    // by splitting columns into nt frames of nspokes columns each
    // then index the frames by an added 4th dimension
    // data is cropped in the process (i.e. some columns might not be used)
    // DON'T REMEMBER IF I DID THIS RIGHT
    mat4DC kdatau = new_mat4DC(nx, nspokes, nc, nt);
    mat3DC ku = new_mat3DC(nx, nspokes, nt);
    mat3D wu = new_mat3D(nx, nspokes, nt);
    for (m = 0; m < nt; m++) {
        for (l = 0; l < nc; l++) {
            for (i = 0; i < nx; i++ {
                for (j = 0; j < nspokes; j++) {
                    kdatau.d[I4D(i,j,l,m,nx,nspokes,nc)] = kdata.d[I3D(i,j*m,l,nx,ntviews)];
                }
             }
        }
    }
    for (l = 0; l < nt; l++) {
        for (i = 0; i < nx; i++ {
            for (j = 0; j < nspokes; j++) {
                ku.d[I3D(i,j,l,nx,nspokes)] = k.d[I2D(i,j*l,ntviews)];
            }
        }
    }
    for (l = 0; l < nt; l++) {
        for (i = 0; i < nx; i++ {
            for (j = 0; j < nspokes; j++) {
                wu.d[I3D(i,j,l,nx,nspokes)] = w.d[I2D(i,j*l,ntviews)];
            }
        }
    }
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
    // %%%%% parameters for reconstruction
    param_type param;
    // param.W = TV_Temp(); (use TV_Temp kernel)
    //param.lambda = 0.25*max(abs(recon_nufft(:)));
    stat = cublasIZamax(handle, b1_total, b1_d, sizeof(cuDoubleComplex), &max_modulus_index);
    const double max_modulus = cuCabs(b1[max_modulus_index]); // cuCabs defined in cuComplex.h
    param.nite = 8;
    param.display = 1;
*/
    // fprintf('\n GRASP reconstruction \n')

    // long starttime = clock_gettime(CLOCK_MONOTONIC, tv_nsec);
    // mat3DC recon_cs=recon_nufft;
    // for (i = 0; i < 4; i++) {
    //     recon_cs = CSL1NlCg(recon_cs,param);
    // }
    // long elapsedtime = (clock_gettime(CLOCK_MONOTONIC, tv_nsec) - starttime)/1000000;

    // recon_nufft=flipdim(recon_nufft,1);
    // recon_cs=flipdim(recon_cs,1);

    // %%%%% display 4 frames
    // recon_nufft2=recon_nufft(:,:,1);
    // recon_nufft2=cat(2,recon_nufft2,recon_nufft(:,:,7));
    // recon_nufft2=cat(2,recon_nufft2,recon_nufft(:,:,13));
    // recon_nufft2=cat(2,recon_nufft2,recon_nufft(:,:,23));
    // recon_cs2=recon_cs(:,:,1);
    // recon_cs2=cat(2,recon_cs2,recon_cs(:,:,7));
    // recon_cs2=cat(2,recon_cs2,recon_cs(:,:,13));
    // recon_cs2=cat(2,recon_cs2,recon_cs(:,:,23));



    // figure;
    // subplot(2,1,1),imshow(abs(recon_nufft2),[]);title('Zero-filled FFT')
    // subplot(2,1,2),imshow(abs(recon_cs2),[]);title('GRASP')



/*
    // get matrix from GPU
    stat = cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
*/

    // free GPU memory
    cudaFree(b1.d);
    cudaFree(k.d);
    cudaFree(kdata.d);
    cudaFree(w.d);

    // destroy cuBLAS context
    cublasDestroy(handle);

    // free CPU memory
    free(b1_cpu);
    free(k_cpu);
    free(kdata_cpu);
    free(w_cpu);
#endif
}
