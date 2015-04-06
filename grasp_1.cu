/*
 * C/CUDA implementation of GRASP.
 * Translated from Matlab by Felix Moody, Fall 2014
 * Dependencies:
 *     CUDA compute capability ????, toolkit version ???, driver version ???
 * Input:
 *	Coil sensitivities b1: (image x, image y, coil)
 *	K-space trajectories: (position in k space,
 * Matrices from liver_data.mat (stored in column major format):
 *     b1: 384x384x12 complex doubles
 *     k: 768x600 complex? doubles
 *     kdata: 768x600x12 complex? doubles
 *     w: 768x600 doubles
 * Data requirements:
 *     1st dim b1 = 2nd dim b1
 *     2nd dim k = 2nd dim kdata = 2nd dim w
 *     3rd dim b1 = 3rd dim kdata
 *     1st dim k = 1st dim kdata =  1st dim w = 2 * 1st dim b1
 * So there are 3 variables to data size: nx, ntviews, and nc, and:
 *     b1 = (nx/2, nx/2, nc)
 *     k = (nx, ntviews)
 *     kdata = (nx, ntviews, nc)
 *     w = (nx, ntviews)
 *
 * GRASP Pipeline:
 * Input:
 *	b1: (amplitude, time, coil)
 *	k:
 *	kdata: (amplitutde, time, coil)
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <cuda_runtime.h>
#include "cublas_v2.h" // CUBLAS header
#include <cuComplex.h> // CUDA complex number and operations header

// Macros to convert 2 and 3 dimensional indices to 1 dimensional row major
#define I2D(i,j,j_tot) = (i*j_tot) + j
#define I3D(i,j,k,i_tot,j_tot) = (i*j_tot) + j + (k*i_tot*j_tot)
#define I4D(i,j,k,l,i_tot,j_tot,k_tot) = (i*j_tot) + j + (k*i_tot*j_tot) + (l*i_tot*j_tot*k_tot)

/*
struct param {
    int nite = 8;
    int display = 1;
    double *** kdatau;
    E // MCNUFFT
    y // undersampled data
    W // Total variate dohicky
    lambda
}
*/
/*
static __inline__ void modify(cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta){
    cublasSscal (handle, n-p, &alpha, &m[IDX2C(p,q,ldm)], ldm);
    cublasSscal (handle, ldm-p, &beta, &m[IDX2C(p,q,ldm)], 1);
}
*/
/*
void objective(x,dx,t,param) {
    //function res = objective(x,dx,t,param) %**********************************

    // %%%%% L2-norm part
    w=param.E*(x+t*dx)-param.y;
    L2Obj=w(:)'*w(:);

    // %%%%% L1-norm part
    if (param.lambda) {
        w = param.W*(x+t*dx);
        L1Obj = sum((conj(w(:)).*w(:)+param.l1Smooth).^(1/2));
    } else {
        L1Obj=0;
    }

    // %%%%% objective function
    res=L2Obj+param.lambda*L1Obj;
}
*/
/*
void grad(x,param) {
    //function g = grad(x,param)%***********************************************

    // %%%%% L2-norm part
    L2Grad = 2.*(param.E'*(param.E*x-param.y));

    // %%%%% L1-norm part
    if (param.lambda) {
        w = param.W*x;
        L1Grad = param.W'*(w.*(w.*conj(w)+param.l1Smooth).^(-0.5));
    } else {
        L1Grad=0;
    }

    // %%%%% composite gradient
    g=L2Grad+param.lambda*L1Grad;
}
*/
/*
void CSL1NlCg(x0, param) {

    % function x = CSL1NlCg(x0,param)
    %
    % res = CSL1NlCg(param)
    %
    % Compressed sensing reconstruction of undersampled k-space MRI data
    %
    % L1-norm minimization using non linear conjugate gradient iterations
    %
    % Given the acquisition model y = E*x, and the sparsifying transform W,
    % the program finds the x that minimizes the following objective function:
    %
    % f(x) = ||E*x - y||^2 + lambda * ||W*x||_1
    %
    % Based on the paper: Sparse MRI: The application of compressed sensing for rapid MR imaging.
    % Lustig M, Donoho D, Pauly JM. Magn Reson Med. 2007 Dec;58(6):1182-95.
    %
    % Ricardo Otazo, NYU 2008
    %

    printf("\n Non-linear conjugate gradient algorithm");
    printf("\n ---------------------------------------------\n");

    // %%%%% starting point
    ????? x = x0;

    // %%%%% line search parameters
    // WHAT TYPES SHOULD THESE ACTUALLY BE?
    int maxlsiter = 150 ;
    double gradToll = 1e-3 ; // does this work?
    double param.l1Smooth = 1e-15;	// ??
    double alpha = 0.01;
    double beta = 0.6;
    double t0 = 1 ;
    double k = 0;

    // %%%%% compute g0  = grad(f(x))
    g0 = grad(x,param);
    dx = -g0;

    // %%%%% iterations
    while(1) {
        // %%%%% backtracking line-search
	f0 = objective(x,dx,0,param);
	t = t0;
        f1 = objective(x,dx,t,param);
	lsiter = 0;
	while (f1 > f0 - alpha*t*abs(g0(:)'*dx(:)))^2 & (lsiter<maxlsiter) {
		lsiter = lsiter + 1;
		t = t * beta;
		f1 = objective(x,dx,t,param);
	}
	if (lsiter == maxlsiter) {
		disp('Error - line search ...');
		return 1;
	}

	// %%%%% control the number of line searches by adapting the initial step search
	if (lsiter > 2), t0 = t0 * beta;end
	if lsiter<1, t0 = t0 / beta; end

        // %%%%% update x
	x = (x + t*dx);

	// %%%%% print some numbers
        if (param.display) {
            fprintf(' ite = %d, cost = %f \n',k,f1);
        }

        // %%%%% conjugate gradient calculation
	g1 = grad(x,param);
	bk = g1(:)'*g1(:)/(g0(:)'*g0(:)+eps);
	g0 = g1;
	dx =  - g1 + bk* dx;
	k = k + 1;

	// %%%%% stopping criteria (to be improved)
	if (k > param.nite) || (norm(dx(:)) < gradToll), break;end

    }
    return;
}
*/
/*
???? MCNUFFT(k,w,b1) {
    // function  res = MCNUFFT(k,w,b1)
    // k and w here are ku and wu in main, which are the columns of k and w split
    // into nt "frames" of nspokes columns, with frames indexed by the added last dimension
    // so, here k is a 768 x nspokes x nt complex double matrix
    // and w is the same sized double matrix

    % Multicoil NUFFT operator
    % Based on the NUFFT toolbox from Jeff Fessler and the single-coil NUFFT
    % operator from Miki Lustig
    % Input
    % k: k-space trajectory
    % w: density compensation
    % b1: coil sensitivity maps
    %
    % Li Feng & Ricardo Otazo, NYU, 2012

    Nd = [nx,ntviews]; // 3rd dim of b1
    Jd = [6,6];
    Kd = [nx*1.5,ntviews*1.5]
    n_shift = [nx, ntviews]Nd/2; // THIS MEANS 3RD DIM OF B1 MUST BE EVEN
    int tt;
    for (tt=1; tt <= nt; tt++) {
        kk=k(:,:,tt); // take the tt'th frame of k
        om = [real(kk(:)), imag(kk(:))]*2*pi; // separate the real and complex components of the frame and save as 1 dim
        res.st{tt} = nufft_init(om, Nd, Jd, Kd, n_shift,'kaiser'); // run nufft
    }
    res.adjoint = 0;
    res.imSize = size(b1(:,:,1));
    res.dataSize = size(k);
    res.w = sqrt(w);
    res.b1 = b1;
    res = class(res,'MCNUFFT');
}
*/

__global__ void elementWiseMultBySqrt(cuDoubleComplex* kdata, double* w) {
    // Definitely not ideal. Is it bad to only use one thread per block?
    // Also we should only have to compute the squares of the elements of w
    // one time and use the result for all slices of kdata
    int i = blockIdx.x + (blockDim.x * blockIdx.y) + (blockDim.x * blockDim.y * threadIdx.x);
    int j = blockIdx.x + (blockDim.x * blockIdx.y);
    cuDoubleComplex sqrtofelement = make_cuDoubleComplex(sqrt(w[j]), 0);
    // possible overflow error with cuCmul (see cuComplex.h)
    kdata[i] = cuCmul(kdata[i], sqrtofelement); // WARNING
}

int main(int argc,char **argv) {
    int i, j, k, l; // general loop indices
    cudaError_t cudaStat; // cuda error type
    cublasStatus_t stat; // CUBLAS error type
    cublasHandle_t handle; // handle to CUBLAS context

    // %%%%%% define number of spokes to be used per frame (Fibonacci number)
    int nspokes = 21;

    // %%%%%% load radial data
    // open matrix files and metadata
    FILE *meta_file = fopen("./liver_data/metadata", "rb");
    FILE *b1_file = fopen("./liver_data/b1.matrix", "rb");
    FILE *k_file = fopen("./liver_data/k.matrix", "rb");
    FILE *kdata_file = fopen("./liver_data/kdata.matrix", "rb");
    FILE *w_file = fopen("./liver_data/w.matrix", "rb");

    // load metadata
    size_t dims[3];
    fread(dims, sizeof(size_t), 3, meta_file);

    // %%%%% data dimensions
    int nx = dims[0];
    int ntviews = dims[1];
    int nc = dims[2];

    // set array total lengths
    int b1_total = nx/2 * nx/2 * nc;
    int k_total = nx * ntviews;
    int kdata_total = nx * ntviews * nc;
    int w_total = nx * ntviews;

    // allocate and load b1, k, kdata, and w on CPU
    cuDoubleComplex * b1 = (cuDoubleComplex *)malloc((nx/2)*(nx/2)*nc * sizeof(cuDoubleComplex));
    fread(b1, sizeof(cuDoubleComplex), (nx/2)*(nx/2)*nc, b1_file);
    cuDoubleComplex * k = (cuDoubleComplex *)malloc(nx*ntviews * sizeof(cuDoubleComplex));
    fread(k, sizeof(cuDoubleComplex), nx*ntviews, k_file);
    cuDoubleComplex * kdata = (cuDoubleComplex *)malloc(nx*ntviews*nc * sizeof(cuDoubleComplex));
    fread(kdata, sizeof(cuDoubleComplex), nx*ntviews*nc, kdata_file);
    double * w = (double *)malloc(nx*ntviews * sizeof(double));
    fread(w, sizeof(double), nx*ntviews, w_file);

    // allocate b1, k, kdata, w on GPU
    cuDoubleComplex * b1_d;
    cuDoubleComplex * k_d;
    cuDoubleComplex * kdata_d;
    double * w_d;
    cudaStat = cudaMalloc((void**)&b1_d, b1_total*sizeof(*b1));
    cudaStat = cudaMalloc((void**)&k_d, kdata_total*sizeof(*k));
    cudaStat = cudaMalloc((void**)&kdata_d, kdata_total*sizeof(*kdata));
    cudaStat = cudaMalloc((void**)&w_d, w_total*sizeof(*w));
    /*if (cudaStat != cudaSuccess |
        cudaStat != cudasuccess |
        cudaStat != cudaSuccess |
        cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }*/

    // copy data from CPU to GPU
    cudaStat = cudaMemcpy(b1_d, b1, sizeof(b1), cudaMemcpyHostToDevice);
    cudaStat = cudaMemcpy(k_d, k, sizeof(k), cudaMemcpyHostToDevice);
    cudaStat = cudaMemcpy(kdata_d, kdata, sizeof(kdata), cudaMemcpyHostToDevice);
    cudaStat = cudaMemcpy(w_d, w, sizeof(w), cudaMemcpyHostToDevice);
    if (cudaStat != cudaSuccess) {
        printf("cudaMemcpy w failed\n");
        return EXIT_FAILURE;
    } else {
        printf("cudaMemcpy w success\n");
    }

    // create cuBLAS context
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    } else {
        printf("CUBLAS initialized\n");
    }

    // b1=b1/max(abs(b1(:)))
    // scale b1 by maximum modulus
    int max_modulus_index;
    stat = cublasIzamax(handle, b1_total, b1_d, sizeof(cuDoubleComplex), &max_modulus_index);
    const double max_modulus = cuCabs(b1[max_modulus_index]); // cuCabs defined in cuComplex.h
    stat = cublasZdscal(handle, b1_total, &max_modulus, b1_d, sizeof(cuDoubleComplex));

    // for ch=1:nc,kdata(:,:,ch)=kdata(:,:,ch).*sqrt(w);endc
    // i.e. multiply each of the 12 slices of kdata element wise by sqrt(w)
    dim3 numBlocks(nx, ntviews);
    elementWiseMultBySqrt<<<numBlocks, nc>>>(kdata_d, w_d);


    // %%%%% number of frames
    int nt = floor(ntviews/nspokes) // do we even need floor here?;

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



    // %%%%% sort the data into a time-series
    // divide the 2nd dimension of kdata, k, and w up into nt "frames" of
    // nspokes columns, then index the frames by an added dimension
    // for ii=1:nt
    //     kdatau(:,:,:,ii)=kdata(:,(ii-1)*nspokes+1:ii*nspokes,:);
    //     ku(:,:,ii)=k(:,(ii-1)*nspokes+1:ii*nspokes);
    //     wu(:,:,ii)=w(:,(ii-1)*nspokes+1:ii*nspokes);
    // end
    cuDoubleComplex * kdatau;
    cuDoubleComplex * ku;
    double * w;

    for(l=0;l < nspokes; l++) {
        for(k=0;k < nc;

/*
    // %%%%% multicoil NUFFT operator
    // param.E=MCNUFFT(ku,wu,b1);
    // USE CUFFT FUNCTION HERE

    // %%%%% undersampled data
    // param.y=kdatau;
    // clear kdata kdatau k ku wu w

    // %%%%% nufft recon
    // recon_nufft=param.E'*param.y;

    // %%%%% parameters for reconstruction
    // param.W = TV_Temp();
    // param.lambda = 0.25*max(abs(recon_nufft(:)));
    // param.nite = 8;
    // param.display = 1;

    // fprintf('\n GRASP reconstruction \n')

    // tic
    // recon_cs=recon_nufft;
    // for n=1:3,
    //     recon_cs = CSL1NlCg(recon_cs,param);
    // end
    // toc

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
*/
/*
    // send matrix to GPU
    stat = cublasSetMatrix (M, N, sizeof(*a), a, M, devPtrA, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    modify (handle, devPtrA, M, N, 1, 2, 16.0f, 12.0f);

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
    cudaFree(b1_d);
    cudaFree(k_d);
    cudaFree(kdata_d);
    cudaFree(w_d);

    // destroy cuBLAS context
    //cublasDestroy(handle);

    // free CPU memory
    free(b1);
    free(k);
    free(kdata);
    free(w);

    // for ch=1:nc,kdata(:,:,ch)=kdata(:,:,ch).*sqrt(w);end
    // this means to multiply each element in each slice of kdata with the
    // square of the corresponding element of w
    //int ch;
    //for(ch = 0; ch < nc; ch++) {
    //    for(
    //    kdata(:,:,ch)=kdata(:,:,ch).*sqrt(w)
    //}
    //printf("%f + %fi\n", creal(b1[3]), cimag(b1[3]));
}
