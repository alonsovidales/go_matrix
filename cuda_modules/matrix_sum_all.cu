/***************************************************
 * Module that applay the function sigmoid to all the elements of the matrix
 * Author: Alonso Vidales <alonso.vidales@tras2.es>
 *
 * To be compiled with nvcc -ptx matrix_sum_all.cu
 * Debug: nvcc -arch=sm_20 -ptx matrix_sum_all.cu
 *
 **************************************************/

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Kernel
__global__ void matrixSumAll(double* A, int wA, int size, double* sum)
{
	__shared__ double res[1024];
	res[threadIdx.x] = 0;

	for (int bx = 0; bx < wA; bx++) {
		int pos = (threadIdx.x * wA) + bx;
		if (pos < size) {
			res[threadIdx.x] += A[pos];
			//printf("Thread %d Pos %d Val: %f\n", threadIdx.x, pos, res[threadIdx.x]);
		}
	}
	__syncthreads();
	if(threadIdx.x == 0) {
		for (int i = 1; i < 1024; i++) {
			res[0] += res[i];
		}
		sum[0] = res[0];
	}
}

#ifdef __cplusplus
}
#endif
