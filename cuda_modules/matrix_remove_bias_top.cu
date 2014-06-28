/***************************************************
 * Module that negs all the elements on a matrix
 * Author: Alonso Vidales <alonso.vidales@tras2.es>
 *
 * To be compiled with nvcc -ptx matrix_remove_bias_top.cu
 * Debug: nvcc -arch=sm_20 -ptx matrix_remove_bias_top.cu
 *
 **************************************************/

//#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Kernel
__global__ void matrixRemoveBiasTop(float* C, float* A, int width, int resW, int resH, int resultSize)
{
	int x = threadIdx.x + (blockIdx.x * resW);
	int y = threadIdx.y + (blockIdx.y * resH);
        int resultPos = y * width + x;

	if (resultPos < resultSize && x <  width) {
		C[resultPos] = A[resultPos + width];
	}
}

#ifdef __cplusplus
}
#endif
