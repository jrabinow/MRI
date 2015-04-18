/*
__global__ void L1HelperKernel(cuDoubleComplex * in, double * out, double l1Smooth) {
    // compute index based on block/grid size
    int i =
    out.d[i] = sqrt(cuCabs(in.d[i]) + l1Smooth);
}

// x and dx are 384x384x28 complex double matrices
double objective(cuDoubleComplex * x, cuDoubleComplex * dx, double t) {
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
mat3DC grad(mat3DC x) {
    // L2-norm part
    // L2Grad =
    // ALLOCATE HERE
    cuDoubleComplex * L2Grad = 2.*(param.E'*(param.E*x-param.y));

    // %%%%% L1-norm part
    if(param.lambda) { // DOES THIS WORK WITH FLOATS?
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
// x0 is a .
mat3DC CSL1NlCg(mat3DC x0, param_type param) {

//  % function x = CSL1NlCg(x0,param)
//  %
//  % res = CSL1NlCg(param)
//  %
//  % Compressed sensing reconstruction of undersampled k-space MRI data
//  %
//  % L1-norm minimization using non linear conjugate gradient iterations
//  %
//  % Given the acquisition model y = E*x, and the sparsifying transform W,
//  % the program finds the x that minimizes the following objective function:
//  %
//  % f(x) = ||E*x - y||^2 + lambda * ||W*x||_1
//  %
//  % Based on the paper: Sparse MRI: The application of compressed sensing for rapid MR imaging.
//  % Lustig M, Donoho D, Pauly JM. Magn Reson Med. 2007 Dec;58(6):1182-95.
//  %
//  % Ricardo Otazo, NYU 2008
//  %

    printf("\n Non-linear conjugate gradient algorithm");
    printf("\n ---------------------------------------------\n");

    // %%%%% starting point
    mat3DC x = copy_mat3DC(x0); // SHOULD I MAKE A COPY OR IS REFERENCE OKAY?

    // %%%%% line search parameters
    int maxlsiter = 150;
    double gradToll = 1e-3;
    param.l1Smooth = 1e-15;
    double alpha = 0.01;
    double beta = 0.6;
    double t0 = 1;
    int k = 0; // iteration counter

    // compute g0  = grad(f(x))
    mat3DC g0 = grad(x);
    mat3DC dx = copy_mat3DC(g0);
    double neg1 = -1.0;
    cublasZdscal(handle, dx.t, &neg1, dx.d, dx.s);


    // %%%%% iterations
    while(1) {
        // %%%%% backtracking line-search
	double f0 = objective(x,dx,0);
	double t = t0;
        double f1 = objective(x,dx,t);
	double lsiter = 0;
        cuDoubleComplex g0dxdotprod;
	while (1) {
                cublasZdotc(handle, g0.t, g0.d, g0.s, dx.d, dx.s, &dotprod);
                if (!(f1 > f0 - alpha*t*cuCabs(dotprod)) || !(lsiter < maxlsiter)) {
                    break;
                }
		lsiter = lsiter + 1.0;
		t = t*beta;
		f1 = objective(x,dx,t);
	}
	if (lsiter == maxlsiter) {
		disp('Error - line search ...');
		return 1;
	}

	// %%%%% control the number of line searches by adapting the initial step search
	if (lsiter > 2) { t0 = t0 * beta; }
	if (lsiter < 1) { t0 = t0 / beta; }

        // %%%%% update x
	// x = (x + t*dx);
        cublasZaxpy(handle, x.t, &make_cuDoubleComplex(t, 0), dx.d, dx.s, x.d, x.s);


	// %%%%% print some numbers
        fprintf("ite = %d, cost = %f\n",k,f1);

        // %%%%% conjugate gradient calculation
	mat3DC g1 = grad(x);
        cuDoubleComplex g1dotprod;
        cuDoubleComplex g0dotprod;
        cublasZdotc(handle, g1.t, g1.d, g1.s, g1.d, g1.s, &g1dotprod);
        cublasZdotc(handle, g0.t, g0.d, g0.s, g0.d, g0.s, &g0dotprod);
        double g1dotprodreal = cuCreal(g1dotprod);
        double g0dotprodreal = cuCreal(g0dotprod);
	double bk = g1dotprodreal/(g0dotprodreal + DBL_EPSILON);
	g0 = g1;
	// dx =  -g1 + bk*dx;
        cublasZdscal(handle, dx.t, &make_cuDoubleComplex(bk, 0.0), dx.d, dx.s);
        cublasZaxpy(handle, g1.t, &neg1,`g1.d, g1.s, dx.d, dx.s);
	k++;

	// %%%%% stopping criteria (to be improved)
        double normdx;
        cublasDznrm2(handle, dx.t, dx.d, dx.s, &normdx);
	if (k > param.nite) || (normdx < gradToll) { break; }
    }
    return x;
}
*/