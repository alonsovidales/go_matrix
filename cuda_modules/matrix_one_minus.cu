/***************************************************
 * Module that negs all the elements on a matrix
 * Author: Alonso Vidales <alonso.vidales@tras2.es>
 *
 * To be compiled with nvcc -ptx matrix_one_minus.cu
 * Debug: nvcc -arch=sm_20 -ptx matrix_one_minus.cu
 *
 **************************************************/

//#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Kernel
__global__ void matrixOneMinus(float* A, int wA, int hA, int width, int finalSize, int matrixSplits)
{
	for (int bx = 0; bx < matrixSplits; bx++) {
		for (int by = 0; by < matrixSplits; by++) {
			int x = threadIdx.x + (bx * wA);
			int y = threadIdx.y + (by * hA);
			int resultPos = y * width + x;

			if (resultPos < finalSize && x < width) {
				A[resultPos] = 1 - A[resultPos];
				//printf("Block %d - %d, thread %d - %d Val: %f\n", x, y, threadIdx.x, threadIdx.y, A[resultPos]);
			}
		}
	}
}

#ifdef __cplusplus
}
#endif
