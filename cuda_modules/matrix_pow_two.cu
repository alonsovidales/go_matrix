/***************************************************
 * Module that applay the function sigmoid to all the elements of the matrix
 * Author: Alonso Vidales <alonso.vidales@tras2.es>
 *
 * To be compiled with nvcc -ptx matrix_pow_two.cu
 * Debug: nvcc -arch=sm_20 -ptx matrix_pow_two.cu
 *
 **************************************************/

//#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Kernel
__global__ void matrixPowTwo(float* A, int wA, int hA, int width, int finalSize, int matrixSplits)
{
	for (int bx = 0; bx < matrixSplits; bx++) {
		for (int by = 0; by < matrixSplits; by++) {
			int x = threadIdx.x + (bx * wA);
			int y = threadIdx.y + (by * hA);
			int resultPos = y * width + x;

			if (resultPos < finalSize && x < width) {
				//printf("IN Block %d - %d, wA: %d thread %d - %d Val: %f resultPos: %d finalSize: %d\n", x, y, wA, threadIdx.x, threadIdx.y, A[resultPos], resultPos, finalSize);
				A[resultPos] = A[resultPos] * A[resultPos];
			}
		}
	}
}

#ifdef __cplusplus
}
#endif
