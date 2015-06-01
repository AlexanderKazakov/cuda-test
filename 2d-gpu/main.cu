#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <sys/time.h>

#include <cuda_runtime.h>

#define BLOCK_DIM 8 

__device__ double c(const double x, const double y) {
	//if ((y > 1.0) && (y <= 1.2)) return 0.8;
	//if ((y > 0.5) && (y <= 0.8) && (x > 0.2) && (x <= 0.5)) return 1.0;
	return 1.0;
}

__device__ double v(const double x, const double t) {
	//if (5*t < 2*M_PI) return sin(5*t) * exp(-30*(x-0.5)*(x-0.5));
	return 1.0;
}

__global__ void calcNodeSimple(double *prev, double *curr, double *next, 
                               const double h_x, const double h_y,
                               const double tau, const double time,
                               const int N, const int M) {
	unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	if ( (i <= N) && (j <= M) ) {
		unsigned int ind = j*(N+1) + i; // 1D-index
		if (j == M) 
			next[ind] = v(i*h_x, time);
		else if (j == 0) 
			next[ind] = curr[ind] 
				  + c(i*h_x, j*h_y) * tau / h_y * (curr[(j+1)*(N+1) + i] - curr[ind]);
		else if (i == 0) 
			next[ind] = curr[ind] 
				  + c(i*h_x, j*h_y) * tau / h_x * (curr[j*(N+1) + i+1] - curr[ind]);
		else if (i == N) 
			next[ind] = curr[ind] 
				  - c(i*h_x, j*h_y) * tau / h_x * (curr[ind] - curr[j*(N+1) + i-1]);
		else  
			next[ind] = 2 * curr[ind] - prev[ind]
				  + c(i*h_x, j*h_y)*c(i*h_x, j*h_y) * tau*tau / (h_x*h_x) * 
				      (curr[j*(N+1) + i+1] - 2*curr[ind] + curr[j*(N+1) + i-1])
				  + c(i*h_x, j*h_y)*c(i*h_x, j*h_y) * tau*tau / (h_y*h_y) * 
				      (curr[(j+1)*(N+1) + i] - 2*curr[ind] + curr[(j-1)*(N+1) + i]);
	}
}

__global__ void calcNode(double *prev, double *curr, double *next, 
                         const double h_x, const double h_y,
                         const double tau, const double time,
                         const int N, const int M) {
	unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (i <= N && j <= M) {
		unsigned int ind = j*(N+1) + i; // 1D-index
		__shared__ double sc[BLOCK_DIM+2][BLOCK_DIM+2];
		double n, u; // values on the next and current time layers in that node
		double p = prev[ind];
		// Copy values of that block to shared memory
		u = sc[threadIdx.x+1][threadIdx.y+1] = curr[ind];

		// Copy necessary values of neighbour blocks to shared memory
		if (threadIdx.x == 0) {
			sc[0][threadIdx.y+1] = 0;
			if (i != 0) sc[0][threadIdx.y+1] = curr[j*(N+1) + i-1];
		}
		if (threadIdx.x == blockDim.x-1) {
			sc[blockDim.x+1][threadIdx.y+1] = 0;
			if (i != N) sc[blockDim.x+1][threadIdx.y+1] = curr[j*(N+1) + i+1];
		}
		if (threadIdx.y == 0) {
			sc[threadIdx.x+1][0] = 0;
			if (j != 0) sc[threadIdx.x+1][0] = curr[(j-1)*(N+1) + i];
		}	
		if (threadIdx.y == blockDim.y-1) {
			sc[threadIdx.x+1][blockDim.y+1] = 0;
			if (j != M) sc[threadIdx.x+1][blockDim.y+1] = curr[(j+1)*(N+1) + i];
		}

		__syncthreads();
		// Calculate next time step
		if (j == M) 
			n = v(i*h_x, time);
		else if (j == 0) 
			n = u + c(i*h_x, j*h_y) * tau / h_y * 
			                             (sc[threadIdx.x+1][threadIdx.y+2] - u);
		else if (i == 0) 
			n = u + c(i*h_x, j*h_y) * tau / h_x * 
			                             (sc[threadIdx.x+2][threadIdx.y+1] - u);
		else if (i == N) 
			n = u - c(i*h_x, j*h_y) * tau / h_x * 
			                               (u - sc[threadIdx.x][threadIdx.y+1]);
		else
			n = 2 * u - p
			  + c(i*h_x, j*h_y)*c(i*h_x, j*h_y) * tau*tau / (h_x*h_x) * 
			      (sc[threadIdx.x+2][threadIdx.y+1] - 2*u + sc[threadIdx.x][threadIdx.y+1])
			  + c(i*h_x, j*h_y)*c(i*h_x, j*h_y) * tau*tau / (h_y*h_y) * 
			      (sc[threadIdx.x+1][threadIdx.y+2] - 2*u + sc[threadIdx.x+1][threadIdx.y]);
		
		// Copy calculated value to global memory
		next[ind] = n;
	}
}

template<typename T>
static void put(std::fstream &f, const T value) {
	union {
		char buf[sizeof(T)];
		T val;
	} helper;
	helper.val = value;
	std::reverse(helper.buf, helper.buf + sizeof(T));
	f.write(helper.buf, sizeof(T));
}

void save(const char *prefix, int step, double *a,
          const double &h_x, const double &h_y,
          const int N, const int M) {
	char buffer[50];
	sprintf(buffer, "%s.%05d.vtk", prefix, step);
	std::fstream f(buffer, std::ios::out);
	if (!f) {
		std::cerr << "Unable to open file " << buffer << std::endl;
		return;
	}
	f << "# vtk DataFile Version 3.0" << std::endl;
	f << "U data" << std::endl;
	f << "BINARY" << std::endl;
	f << "DATASET STRUCTURED_POINTS" << std::endl;
	f << "DIMENSIONS " << N+1 << " " << M+1 << " 1" << std::endl;
	f << "SPACING " << h_x << " " << h_y << " 1" << std::endl;
	f << "ORIGIN 0 0 0" << std::endl;
	f << "POINT_DATA " << (N+1) * (M+1) << std::endl;
	f << "SCALARS u double" << std::endl;
	f << "LOOKUP_TABLE default" << std::endl;
	for (int j = 0 ; j < M+1; j++){
		for (int i = 0; i < N+1; i++)
			put(f, a[j*(N+1) + i]);
	}
	f.close();
}

/**
 * Calculate the process, save results.
 */
void calculate(double *prev, double *curr, double *next,
               double *hostData, const int N, const int M) {
	const double h_x = 0.01;
	const double h_y = 0.005;
	const double tau = h_y / 2.0;
	const double T = 5.0;
	double curTime = 0.0;
	int counter = 0;
	double visualisationStep = T / 100;
	float maxTime = 0;
	float avgTime = 0;
	float minTime = 9e+9;
	std::cout << "Max time of kernel execution:\nstep\ttime\n";
	while (curTime <= T) {
		dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);
		dim3 numBlocks(N/threadsPerBlock.x + 1, M/threadsPerBlock.y + 1); 
		
		// Cuda Events for measuring of execution time
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		
		cudaEventRecord(start);
		calcNode <<<numBlocks, threadsPerBlock>>> 
		                    (prev, curr, next, h_x, h_y, tau, curTime, N, M);
		cudaEventRecord(stop);
		
		// Evaluating execution time
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		if (maxTime < milliseconds)
			std::cout << curTime/tau << "\t" << milliseconds << "\n";
		maxTime = (maxTime > milliseconds) ? maxTime : milliseconds;
		minTime = (minTime < milliseconds) ? minTime : milliseconds;
		avgTime += milliseconds / (T / tau);
		
		double *tmp = prev;
		prev = curr;
		curr = next;
		next = tmp;

		if ( fabs(curTime - counter * visualisationStep) < 1e-5 ) {
			cudaError_t err = cudaMemcpy(hostData, prev, 
			                             (N+1) * (M+1) * sizeof (double),
			                             cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) {
				fprintf(stderr, "Failed to copy data from device to host",
					"(error code %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}

			save("output", counter, hostData, h_x, h_y, N, M);
			counter++;
		}
		curTime += tau;
 	}
	std::cout << "Kernel execution in milliseconds:\nmax time = " << maxTime << 
	             "\navg time = " << avgTime << "\nmin time = " << minTime << "\n";
}

double mtime()
{
  struct timeval t;

  gettimeofday(&t, NULL);
  double mt = (double) (t.tv_sec) * 1000 + (double) t.tv_usec / 1000;
  return mt;
}

int main(int argc, char **argv) {
	const int N = 1000;
	const int M = 4000;
	size_t size = (N+1) * (M+1) * sizeof (double);
	cudaError_t err = cudaSuccess;
	
	double *prev = NULL;
	err = cudaMalloc((void **) &prev, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device memory!\n",
		        cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	double *curr = NULL;	
	err = cudaMalloc((void **) &curr, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device memory!\n",
		        cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	double *next = NULL;	
	err = cudaMalloc((void **) &next, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device memory!\n",
		        cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	double *hostData = new double[(N+1)*(M+1)];

	cudaMemset(prev, 0, size);	
	cudaMemset(curr, 0, size);	
	cudaMemset(next, 0, size);	
	memset(hostData, 0, size);
	
	double t1 = mtime();
	calculate(prev, curr, next, hostData, N, M);
	double t2 = mtime();
	std::cout << "Time of calculate() execution = " << t2 - t1 << std::endl;
	
	cudaFree(prev);
	cudaFree(curr);
	cudaFree(next);
	delete [] hostData;
}
