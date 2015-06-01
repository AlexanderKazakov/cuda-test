#include <stdio.h>
#include <iostream>
#include <ctime>
#include <cmath>
#include <fstream>
#include <unistd.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements. Each thread adds k elements of vectors.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C,
                          const int numElements, const int k) {
	int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
	for (int j = 0; j < k; j++) {
		int i = threadIndex * k + j;
		if (i < numElements) C[i] = A[i] + B[i];
	}
}

int main(int argc, char** argv) {
	int numElements = 0;
	int k = 1;
	int opt = 0;
	while ((opt = getopt(argc, argv, "n:k:")) != -1) {
		switch (opt) {
			case 'n': numElements = atoi(optarg);
				break;
			case 'k': k = atoi(optarg);
				break;
			case '?':
			{
				fprintf(stderr, "Usage: ./addVector -n numElementsToAdd",
				                " -k numElementsToAddInOneThread\n");
				exit(-1);
			}
		}
	}
	if ((numElements < 1) || (k < 1)) {
		fprintf(stderr, "Bad parameters!\n");
		exit(-1);
	}
	printf("Adding vectors of size %d ...\n", numElements);
	
	// Allocate the host input vectors
	size_t size = numElements * sizeof (float);
	float *h_A = (float *) malloc(size);
	float *h_B = (float *) malloc(size);
	float *h_C = (float *) malloc(size);
	if (h_A == NULL || h_B == NULL || h_C == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}
	// Initialize the host input vectors
	srand(time(0));
	for (int i = 0; i < numElements; ++i) {
		h_A[i] = rand() / (float) RAND_MAX;
		h_B[i] = rand() / (float) RAND_MAX;
	}

	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;
	
	// Allocate the device input vectors
	float *d_A = NULL;
	err = cudaMalloc((void **) &d_A, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
		        cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	float *d_B = NULL;
	err = cudaMalloc((void **) &d_B, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
		        cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	float *d_C = NULL;
	err = cudaMalloc((void **) &d_C, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
		        cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the host input vectors A and B in host memory 
	// to the device input vectors in device memory
	printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n",
		        cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", 
		        cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 512;
	int blocksPerGrid = ((numElements + k - 1) / k + threadsPerBlock - 1) / threadsPerBlock;

	// Cuda Events for measuring of execution time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	printf("CUDA kernel launch with %d blocks of %d threads\n", 
	       blocksPerGrid, threadsPerBlock);

	cudaEventRecord(start);
	vectorAdd <<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements, k);
	cudaEventRecord(stop);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
		        cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Evaluating execution time
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	// Evaluating computational bandwidth in GFLOPs
	float bandwidth = numElements / milliseconds / 1e+6;

	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy vector C from device to host",
		                "(error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Verify that the result vector is correct
	for (int i = numElements - 5; i < numElements; ++i) {
		if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}
	srand(time(0));
	for (int counter = 0; counter < 10; counter++) {
		int i = (int) ( (rand() / (float) RAND_MAX) * numElements );
		if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}

	// Free device global memory
	err = cudaFree(d_A);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", 
		        cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaFree(d_B);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to free device vector B (error code %s)!\n", 
		        cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaFree(d_C);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to free device vector C (error code %s)!\n", 
		        cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);

	// Reset the device and exit
	err = cudaDeviceReset();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to deinitialize the device! error=%s\n", 
		        cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Write performance data to file for performance plot
	std::ofstream performanceData;
	performanceData.open("performanceData.txt", std::ios::app);
	performanceData << log10(numElements) << "\t" << bandwidth << "\n";
	performanceData.close();

	printf("Done in %f milliseconds with computational performance %f GFLOPs\n",
	       milliseconds, bandwidth);
	return 0;
}

