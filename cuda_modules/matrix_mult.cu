/***************************************************
 * Module for matrix multiplication
 * Author: Alonso Vidales <alonso.vidales@tras2.es>
 *
 * To be compiled with nvcc -ptx matrix_mult.cu
 * Debug: nvcc -arch=sm_20 -ptx matrix_mult.cu
 *
 **************************************************/

//#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Kernel
__global__ void matrixMul(float* C, float* A, float* B, int wA, int wB, int resW, int resH, int resultSize, int matrixSplits)
{
	for (int bx = 0; bx < matrixSplits; bx++) {
		for (int by = 0; by < matrixSplits; by++) {
			int x = threadIdx.x + (bx * resW);
			int y = threadIdx.y + (by * resH);
			int resultPos = y * wB + x;
			//printf("Block %d - %d / %d - %d thread %d - %d Pos: %d - C: %d\n", x, y, bx, by, threadIdx.x, threadIdx.y, resultPos, resultSize);

			//printf("Thread %d - %d: %d. Final: x: %d y: %d\n", threadIdx.x, threadIdx.y, resultPos, x, y);
			if (x < wB && resultPos < resultSize) {
				// value stores the element that is 
				// computed by the thread
				float value = 0;
				for (int i = 0; i < wA; ++i)
				{
					value += A[y * wA + i] * B[i * wB + x];
				}

				// Write the matrix to device memory each 
				// thread writes one element
				C[resultPos] = value;
	
				//printf("Block %d - %d, thread %d - %d : Width, %d Pos: %d Val: %f\n", x, y, threadIdx.x, threadIdx.y, wB, resultPos, value);
			}
		}
	}
}

#ifdef __cplusplus
}
#endif
