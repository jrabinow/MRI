 /*
 * C/CUDA implementation of GRASP. 
 * Translated from Matlab by Felix Moody, Fall 2014  
 * Dependencies:
 *     CUDA compute capability ????, toolkit version ???, driver version ???
 *     Developed on Tesla T10 (see "CIMS cuda3 deviceQuery output.txt")
 * To compile with nvcc and cublas
 *     $nvcc grasp.cu -o grasp -lcublas
 * Input: from liver_data.mat (stored in column major format): 
 *     Coil sensitivities b1 -- 384x384x12 complex doubles (image x, image y, coil)
 *     K-space trajectories k -- 768x600 complex doubles (position in k space, experiment)
 *     Sample density compensation w: 768x600 doubles (real number between 0 and 1, experiment)
 *     Experimental data kdata: 768x600x12 complex doubles (k space reading, experiment, coil)
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
 * Todo/Questions (also caps means questions):
 *     Could we and should we use CUSPARSE instead of cuBLAS? 
 *     Can multiple threads access the same memory? At the same time?
 *     How do we think about blocks vs grids? When is it best to break into
 *        blocks and when grids if data fits both?
 *     Should param be global or passed to each subfunction?
 *     Should I break subfunctions into separate files? Why/Why not? How?
 *     Is it worth the space/time to have matrix structs and constructors?
 *         Definitely makes sense to have cudaMalloc packaged into function
 *         Maybe make it an inline function?
 *     In how much generality should we code? For example, should we assume
 *         a certain data size for optimization, or data type?
 *     How to handle errors?
 *     Must the program be designed with a specific data size in mind?
 *     Can we automatically optimize GPU given any data size (in some range)
 *     Not sure if I pulled data from liver_data.mat correctly
 *         
 * Notes
 *     Any strange complex number stuff is defined in cuComplex.h
 *     Unless otherwise stated, all matlab variables are doubles
 *     %%%%%% = comments from matlab
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h> // for square root
#include <complex.h> // C complex numbers and operations; DO I EVEN USE THIS?
#include <cuda_runtime.h>
#include "cublas_v2.h" // CUBLAS
#include <cuComplex.h> // CUDA complex numbers and operations

// Macros to convert multidimensional indices to 1 dimensional row major index
#define I2D(i,j,j_tot) = (i*j_tot) + j
#define I3D(i,j,k,i_tot,j_tot) = (i*j_tot) + j + (k*i_tot*j_tot)
#define I4D(i,j,k,l,i_tot,j_tot,k_tot) = (i*j_tot) + j + (k*i_tot*j_tot) + (l*i_tot*j_tot*k_tot)

// matrix types
struct mat2D {
    double * d; // the actual data
    int x; // 1st dim size
    int y; // 2nd dim size
    int t; // total # of entries
    int s; // size in bytes of each entry
};
struct mat2DC {
    cuDoubleComplex * d;
    int x; // 1st dim size
    int y; // 2nd dim size
    int t;
    int s; 
};
struct mat3D {
    double * d; 
    int x; // 1st dim size
    int y; // 2nd dim size
    int z; // 3rd dim size
    int t;
    int s;
};
struct mat3DC {
    cuDoubleComplex * d;
    int x;
    int y;
    int z;
    int t;
    int s;
};
struct mat4D {
    double * d;
    int x; // 1st dim size
    int y; // 2nd dim size
    int z; // 3rd dim size 
    int w; // 4th dim size
    int t;
    int s;
};
struct mat4DC {
    cuDoubleComplex * d;
    int x;
    int y;
    int z;
    int w;
    int t;
    int s;
};

// "constructors" for matrix types
mat2D new_mat2D(int xsize, int ysize) {
    mat2D thismat;
    thismat.x = xsize;
    thismat.y = ysize;
    thismat.t = xsize*ysize;
    thismat.s = sizeof(double);
    cudaError_t err = cudaMalloc((void**)&(thismat.d), (thismat.t)*(thismat.s));
    if (err != cudaSuccess) { 
        fprintf(stderr, "Failed allocating matrix on GPU\n");
        exit(EXIT_FAILURE);
    }
    return thismat;
}
mat2DC new_mat2DC(int xsize, int ysize) {
    mat2DC thismat;
    thismat.x = xsize;
    thismat.y = ysize;
    thismat.t = xsize*ysize;
    thismat.s = sizeof(cuDoubleComplex);
    cudaError_t err = cudaMalloc((void**)&(thismat.d), (thismat.t)*(thismat.s));
    if (err != cudaSuccess) { 
        fprintf(stderr, "Failed allocating matrix on GPU\n");
        exit(EXIT_FAILURE);
    }
    return thismat;
}
mat3D new_mat3D(int xsize, int ysize, int zsize) {
    mat3D thismat;
    thismat.x = xsize;
    thismat.y = ysize;
    thismat.z = zsize;
    thismat.t = xsize*ysize*zsize;
    thismat.s = sizeof(double);
    cudaError_t err = cudaMalloc((void**)&(thismat.d), (thismat.t)*(thismat.s));
    if (err != cudaSuccess) { 
        fprintf(stderr, "Failed allocating matrix on GPU\n");
        exit(EXIT_FAILURE);
    }
    return thismat;
}
mat3DC new_mat3DC(int xsize, int ysize, int zsize) {
    mat3DC thismat;
    thismat.x = xsize;
    thismat.y = ysize;
    thismat.z = zsize;
    thismat.t = xsize*ysize*zsize;
    thismat.s = sizeof(cuDoubleComplex);
    cudaError_t err = cudaMalloc((void**)&(thismat.d), (thismat.t)*(thismat.s));
    if (err != cudaSuccess) { 
        fprintf(stderr, "Failed allocating matrix on GPU\n");
        exit(EXIT_FAILURE);
    }
    return thismat;
}
mat4D new_mat4D(int xsize, int ysize, int zsize, int wsize) {
    mat4D thismat;
    thismat.x = xsize;
    thismat.y = ysize;
    thismat.z = zsize;
    thismat.w = wsize;
    thismat.t = xsize*ysize*zsize*wsize;
    thismat.s = sizeof(double);
    cudaError_t err = cudaMalloc((void**)&(thismat.d), (thismat.t)*(thismat.s));
    if (err != cudaSuccess) { 
        fprintf(stderr, "Failed allocating matrix on GPU\n");
        exit(EXIT_FAILURE);
    }
    return thismat;
}
mat4DC new_mat4DC(int xsize, int ysize, int zsize, int wsize) {
    mat4DC thismat;
    thismat.x = xsize;
    thismat.y = ysize;
    thismat.z = zsize;
    thismat.w = wsize;
    thismat.t = xsize*ysize*zsize*wsize;
    thismat.s = sizeof(cuDoubleComplex);
    cudaError_t err = cudaMalloc((void**)&(thismat.d), (thismat.t)*(thismat.s));
    if (err != cudaSuccess) { 
        fprintf(stderr, "Failed allocating matrix on GPU\n");
        exit(EXIT_FAILURE);
    }
    return thismat;
}

// matrix duplicate functions
mat3D copy_mat3D(mat3D in) {
    mat3D thismat = new_mat3D(in.x, in.y, in.z);
    cudaError_t err = cudaMemcpy(thismat.d, in.d, in.t*in.s, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) { 
        fprintf(stderr, "Failed duplicating matrix on GPU\n");
        exit(EXIT_FAILURE);
    }
    return thismat;
}
mat3DC copy_mat3DC(mat3DC in) {
    mat3DC thismat = new_mat3DC(in.x, in.y, in.z);
    cudaError_t err = cudaMemcpy(thismat.d, in.d, in.t*in.s, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) { 
        fprintf(stderr, "Failed duplicating matrix on GPU\n");
        exit(EXIT_FAILURE);
    }
    return thismat;
}
    
/*
struct param_type { // ARE THESE THE RIGHT TYPES?
    cuDoubleComplex * E; // MCNUFFT
    cuDoubleComplex * y; // kdatau
    cuDoubleComplex * W; // Total variate dohicky
    double lambda; // trade off control (BETWEEN WHAT?)
    double l1Smooth; // WHAT DOES THIS DO?
    int nite = 8; //
} param;
*/


/*
static __inline__ void modify(cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta){
    cublasSscal (handle, n-p, &alpha, &m[IDX2C(p,q,ldm)], ldm);
    cublasSscal (handle, ldm-p, &beta, &m[IDX2C(p,q,ldm)], 1);
}
*/

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



/*

__global__ void L1HelperKernel(cuDoubleComplex * in, double * out, double l1Smooth) {
    // compute index based on block/grid size
    int i = 
    out.d[i] = sqrt(cuCabs(in.d[i]) + l1Smooth);
}

// x and dx are 384x384x28 complex double matrices
void objective(cuDoubleComplex * x, cuDoubleComplex * dx, double t) {
    //function res = objective(x,dx,t,param) %**********************************

    // %%%%% L2-norm part
    // w = param.E*(x+t*dx)-param.y;
    // L2Obj=w(:)'*w(:)

    // cast scalars for cuBLAS compatibility
    cuDoubleComplex t_complex = make_cuDoubleComplex(t,(double)0);
    cuDoubleComplex minus1 = make_cuDoubleComplex((double)-1,(double)0);
    // copy x so it doesn't get overwritten
    mat3DC next_x copy_mat3DC(x);
    // next_x=x+t*dx
    cublasZaxpy(handle, x.t, &t_complex, dx.d, dx.s, next_x.d, next_x.s);
    // INSERT FFT HERE
    // mat3DC ft = MCNUFFT(next_x);
    //  ft = ft + (-1)*param.y 
    cublasZaxpy(handle, x.t, &minus1, param.y.d, param.y.s, ft.d, ft.s);
    // L2Obj = ft complex dot product ft
    cuDoubleComplex L2Obj;
    cublasZdotc(handle, ft.t, ft.s, ft.t, ft.s, &L2Obj); // IS THIS RIGHT? 

    // %%%%% L1-norm part
    // w = param.W*(x+t*dx);
    // L1Obj = sum((conj(w(:)).*w(:)+param.l1Smooth).^(1/2));
    // In matlab code L1Obj wasn't calculated if lambda=0
    mat3DC w = new_mat3DC(next_x.x, next_x.y, next_x.z);
    TV_temp(next_x.d, w.d, 0);
    mat3DC temp = new_mat3D(w.x, w.y, w.z);
    dim3 numBlocks(w.x, w.y);
    L1HelperKernel<<numBlocks, w.z>>(w, temp, param.l1Smooth);
    double L1Obj;
    cublasDasum(handle, temp.t, temp.d, temp.s, &L1Obj);
    
    // %%%%% objective function
    return L2Obj+param.lambda*L1Obj;
}
*/
/*
cuDoubleComplex * grad(cuDoubleComplex x, param_type param) {
    // L2-norm part
    // L2Grad =
    // ALLOCATE HERE
    cuDoubleComplex * L2Grad = 2.*(param.E'*(param.E*x-param.y));

    // %%%%% L1-norm part
    if (param.lambda) { // DOES THIS WORK WITH FLOATS?
        // ALLOCATE HERE
        cuDoubleComplex w = param.W*x;
        // v RIGHT TYPE? ALLOCATE
        cuDoubleComplex L1Grad = param.W'*(w.*(w.*conj(w)+param.l1Smooth).^(-0.5));
    } else { // no need to calculate L1Grad if 0 lambda value nullifies it
        return L2Grad;
    }

    //SCALE L1Grad BY LAMBDA WITH CUBLAS FUNCTION

    // %%%%% composite gradient
    return L2Grad+param.lambda*L1Grad;
}
*/

/*
// x0 is a 384x384x28 complex double matrix.
void CSL1NlCg(cuDoubleComplex x0, param_type param) {

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

    // starting point
    cuDoubleComplex * x = x0; // SHOULD I MAKE A COPY OR IS REFERENCE OKAY?

    // line search parameters
    // WHAT TYPES SHOULD THESE ACTUALLY BE?
    int maxlsiter = 150;
    double gradToll = 1e-3;
    param.l1Smooth = 1e-15;
    double alpha = 0.01;  
    double beta = 0.6;
    double t0 = 1; 
    double k = 0;

    // compute g0  = grad(f(x))
    cuDoubleComplex * g0 = grad(x,param);
    cuDoubleComplex * dx = -g0; // USE CUBLAS FUNCTION

    // %%%%% iterations
    while(1) {
        // %%%%% backtracking line-search
	TYPE? f0 = objective(x,dx,0,param);
	double t = t0;
        TYPE? f1 = objective(x,dx,t,param);
	TYPE? lsiter = 0;
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
    // Data size (determines gpu optimization, so don't change lightly!)
    int nx = 768;
    int ntviews = 600;
    int nc = 12;

    int i, j, l, m; // general loop indices (skipped k due to conflict)   
    cudaError_t cudaStat; // cuda error variable
    cublasStatus_t stat; // CUBLAS error variable
    cublasHandle_t handle; // handle to CUBLAS context

    // %%%%%% define number of spokes to be used per frame (Fibonacci number)
    int nspokes = 21;

    // %%%%%% load radial data
    // open matrix files and metadata
    // these were pulled from liver_data.mat by convertmat
    FILE * meta_file = fopen("./liver_data/metadata", "rb");
    FILE * b1_file = fopen("./liver_data/b1.matrix", "rb");
    FILE * k_file = fopen("./liver_data/k.matrix", "rb");
    FILE * kdata_file = fopen("./liver_data/kdata.matrix", "rb");
    FILE * w_file = fopen("./liver_data/w.matrix", "rb");

/*  This section was superceded by the constant data size defined above,
    and the use of matrix structs and constructors
    // load metadata
    size_t dims[3];
    fread(dims, sizeof(size_t), 3, meta_file);

    // %%%%% data dimensions
    //int nx = dims[0];
    //int ntviews = dims[1];
    //int nc = dims[2];

    // set array total lengths
    //int b1_total = nx/2 * nx/2 * nc;
    //int k_total = nx * ntviews;
    //int kdata_total = nx * ntviews * nc;
    //int w_total = nx * ntviews;
*/

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
   
    /* (replaced by matrix structs and constructors
    cuDoubleComplex * b1_d;
    cuDoubleComplex * k_d;
    cuDoubleComplex * kdata_d;
    double * w_d;
    cudaStat = cudaMalloc((void**)&b1_d, b1_total*sizeof(*b1));
    cudaStat = cudaMalloc((void**)&k_d, kdata_total*sizeof(*k));
    cudaStat = cudaMalloc((void**)&kdata_d, kdata_total*sizeof(*kdata));
    cudaStat = cudaMalloc((void**)&w_d, w_total*sizeof(*w));
    if (cudaStat != cudaSuccess |
        cudaStat != cudasuccess |
        cudaStat != cudaSuccess |
        cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }*/

    // copy data from CPU to GPU
    cudaStat = cudaMemcpy(b1.d, b1_cpu, b1.s*b1.t, cudaMemcpyHostToDevice);
    if (cudaStat == cudaSuccess) {
        cudaMemcpy(k.d, k_cpu, k.s*k.t, cudaMemcpyHostToDevice);
    }
    if (cudaStat == cudaSuccess) {
        cudaMemcpy(kdata.d, kdata_cpu, kdata.s*kdata.t, cudaMemcpyHostToDevice);
    }
    if (cudaStat == cudaSuccess) {
        cudaMemcpy(w.d, w_cpu, w.s*w.t, cudaMemcpyHostToDevice);
    }
    if (cudaStat != cudaSuccess) {
        fprintf(stderr, "Failed copying data from CPU to GPU\n");
        exit(EXIT_FAILURE);
    } else {
        printf("Copied data from CPU to GPU\n");
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
    stat = cublasIzamax(handle, b1.t, b1.d, b1.s, &max_modulus_index);
    const double max_modulus = cuCabs(b1_cpu[max_modulus_index]);
    stat = cublasZdscal(handle, b1.t, &max_modulus, b1.d, b1.s);
/*
    // for ch=1:nc,kdata(:,:,ch)=kdata(:,:,ch).*sqrt(w);endc
    // i.e. multiply each of the 12 slices of kdata element-wise by sqrt(w)
    dim3 numBlocks(nx, ntviews);
    elementWiseMultBySqrt<<<numBlocks, nc>>>(kdata.d, w.d);

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
    recon_nufft=param.E'*param.y;
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
}
