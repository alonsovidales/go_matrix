/***************************************************
 * Module that negs all the elements on a matrix
 * Author: Alonso Vidales <alonso.vidales@tras2.es>
 *
 * To be compiled with nvcc -ptx matrix_add_bias.cu
 * Debug: nvcc -arch=sm_20 -ptx matrix_add_bias.cu
 *
 **************************************************/

//#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Kernel
__global__ void matrixAddBias(float* C, float* A, int wA, int hA, int width, int finalSize, int matrixSplits)
{
	for (int bx = 0; bx < matrixSplits; bx++) {
		for (int by = 0; by < matrixSplits; by++) {
			int x = threadIdx.x + (bx * wA);
			int y = threadIdx.y + (by * hA);
			int resultPos = y * width + x;

			if (resultPos < finalSize && x <  width) {
				if (x == 0) {
					C[resultPos] = 1;
					//printf("Block %d - %d, thread %d - %d Val: %f Pos: %d Row: ONE\n", x, y, threadIdx.x, threadIdx.y, C[resultPos], resultPos);
				} else {
					C[resultPos] = A[resultPos - (resultPos / width + 1)];
					//printf("Block %d - %d, thread %d - %d Val: %f Pos: %d Row: %d\n", x, y, threadIdx.x, threadIdx.y, C[resultPos], resultPos, resultPos - (resultPos / width + 1));
				}
			}
		}
	}
}

#ifdef __cplusplus
}
#endif
