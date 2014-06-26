/***************************************************
 * Module that multiply a matrix by the transpose of other
 * Author: Alonso Vidales <alonso.vidales@tras2.es>
 *
 * To be compiled with nvcc -ptx matrix_mult_trans.cu
 * Debug: nvcc -arch=sm_20 -ptx matrix_mult_trans.cu
 *
 **************************************************/

//#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Kernel
__global__ void matrixMulTrans(float* C, float* A, float* B, int wA, int resRealW, int resW, int resH, int resultSize, int matrixSplits)
{
	for (int bx = 0; bx < matrixSplits; bx++) {
		for (int by = 0; by < matrixSplits; by++) {
			int x = threadIdx.x + (bx * resW);
			int y = threadIdx.y + (by * resH);
			int resultPos = y * resRealW + x;

			//printf("Thread %d - %d: %d. Final: x: %d y: %d Size: %d\n", threadIdx.x, threadIdx.y, resultPos, x, y, resultSize);
			if ((resultPos < resultSize) && (x < resRealW)) {
				// value stores the element that is 
				// computed by the thread
				float value = 0;
				for (int i = 0; i < wA; ++i)
				{
					value += A[y * wA + i] * B[x * wA + i];
				}

				// Write the matrix to device memory each 
				// thread writes one element
				C[resultPos] = value;
	
				//printf("Block %d - %d, thread %d - %d : Pos: %d Val: %f\n", x, y, threadIdx.x, threadIdx.y, resultPos, value);
			}
		}
	}
}

#ifdef __cplusplus
}
#endif
