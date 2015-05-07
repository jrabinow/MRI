
MRI
===

#### Notes

- Any strange complex number stuff is defined in cuComplex.h

- Dynamically linked. To run the program, you must run as follows:
$ LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./gpuNUFFT-2.0.6rc2/CUDA/bin ./grasp


#### TODO

- Could we and should we use CUSPARSE instead of cuBLAS?
- Can multiple threads access the same memory? At the same time?
- How do we think about blocks vs grids? When is it best to break into
blocks and when grids if data fits both?
- Should param be global or passed to each subfunction?
- Should I break subfunctions into separate files? Why/Why not? How?
- In how much generality should we code? For example, should we assume
a certain data size for optimization, or data type?
- How to handle errors?
- Must the program be designed with a specific data size in mind?
- Can we automatically optimize GPU given any data size (in some range)
- Not sure if I pulled data from liver_data.mat correctly
- In cublas, is it better to take the dot product of a vector with
itself, or to take the norm and then square it?


#### CUDA Programming Notes

-For compute capability < 3.5, cuBLAS functions can only be called from
the host
-Where to store metadata? cuBLAS functions must be called from CPU,
so no benefit to storing metadata on device 


#### Dependencies

- CUDA compute capability ????, toolkit version ???, driver version ???
- Developed on Tesla T10 (see "CIMS cuda3 deviceQuery output.txt")


### Input

From liver_data.mat (stored in column major format):
- Coil sensitivities b1
384x384x12 complex doubles (image x, image y, coil)
- K-space trajectories k
768x600 complex doubles (position in k space, experiment)
- Sample density compensation w
768x600 doubles (real number between 0 and 1, experiment)
- Experimental data kdata
768x600x12 complex doubles (k space reading, experiment, coil)

So there are 3 variables to data size: nx, ntviews, and nc, and:
- b1 = (nx/2, nx/2, nc)
- k = (nx, ntviews)
- kdata = (nx, ntviews, nc)
- w = (nx, ntviews)


#### Matlab Notes

- Unless otherwise stated, all matlab variables are doubles
