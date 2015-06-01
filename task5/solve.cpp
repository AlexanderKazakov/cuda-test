#include <complex>
#include "cuComplex.h"
#include <cuda_runtime.h>
#include <cusparse.h>

using namespace std;

int solve(int m, complex<float> *top_h, complex<float> *mid_h, 
                 complex<float> *bot_h, complex<float> *b_h, 
                 complex<float> *x_h) {
	
	cusparseStatus_t status;
	cusparseHandle_t handle = 0;
	status = cusparseCreate(&handle);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		fprintf(stderr, "Failed to init cuSPARSE library!\n");
		return (-1);
	}

	cuComplex *top_d, *mid_d, *bot_d, *b_d;
	int size = sizeof(cuComplex) * m;
	cudaMalloc((void**)&top_d, size);
	cudaMalloc((void**)&mid_d, size);
	cudaMalloc((void**)&bot_d, size);
	cudaMalloc((void**)&b_d, size);

	cudaMemcpy(top_d, top_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(mid_d, mid_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(bot_d, bot_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

	/* http://docs.nvidia.com/cuda/cusparse/#cusparse-lt-t-gt-gtsv */
	status = cusparseCgtsv(handle, m, 1, bot_d, mid_d, top_d, b_d, m);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		fprintf(stderr, "Failed to solve SLE by cuSPARSE library!\n");
		return(-2);
	}

	cudaMemcpy(x_h, b_d, size, cudaMemcpyDeviceToHost);

	cudaFree(top_d);
	cudaFree(mid_d);
	cudaFree(bot_d);
	cudaFree(b_d);

	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		fprintf(stderr, "The last cuda error is %s", cudaGetErrorString(cudaError));
		return(-3);
	}

	return 0;
}

