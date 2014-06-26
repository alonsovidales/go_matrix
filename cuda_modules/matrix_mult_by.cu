/***************************************************
 * Module that multiply all the elements of a matrix by a number
 * Author: Alonso Vidales <alonso.vidales@tras2.es>
 *
 * To be compiled with nvcc -ptx matrix_mult_by.cu
 * Debug: nvcc -arch=sm_20 -ptx matrix_mult_by.cu
 *
 **************************************************/

//#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Kernel
__global__ void matrixMultBy(float* A, float multBy, int wA, int hA, int width, int finalSize, int matrixSplits)
{
	for (int bx = 0; bx < matrixSplits; bx++) {
		for (int by = 0; by < matrixSplits; by++) {
			int x = threadIdx.x + (bx * wA);
			int y = threadIdx.y + (by * hA);
			int resultPos = y * width + x;

			if (resultPos < finalSize && x <  width) {
				A[resultPos] *= multBy;
				//printf("Block %d - %d, thread %d - %d Val: %f\n", x, y, threadIdx.x, threadIdx.y, A[resultPos]);
			}
		}
	}
}

#ifdef __cplusplus
}
#endif
