/***************************************************
 * Module for matrix substraction
 * Author: Alonso Vidales <alonso.vidales@tras2.es>
 *
 * To be compiled with nvcc -ptx matrix_sub.cu
 * Debug: nvcc -arch=sm_20 -ptx matrix_sub.cu
 *
 **************************************************/

//#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Kernel
__global__ void matrixSub(float* C, float* A, float* B, int wA, int resW, int resH, int resultSize, int matrixSplits)
{
	for (int bx = 0; bx < matrixSplits; bx++) {
		for (int by = 0; by < matrixSplits; by++) {
			int x = threadIdx.x + (bx * resW);
			int y = threadIdx.y + (by * resH);
			int resultPos = y * wA + x;

			if (resultPos < resultSize && x < wA) {
				C[resultPos] = A[resultPos] - B[resultPos];
				//printf("Block %d - %d, thread %d - %d Val: %f\n", x, y, threadIdx.x, threadIdx.y, C[resultPos]);
			}
		}
	}
}

#ifdef __cplusplus
}
#endif
