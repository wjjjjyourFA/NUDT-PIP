#ifndef Merge
#define Merge
#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda.h>
#define MMAX(a, b)(a > b ? a : b)
#define MMIN(a, b)(a < b ? a : b)

extern "C"
{
    void MergeColor(const float* disp_map, int height, int width, int max_area, float *output);
}
__global__ void Merge_kernel(const float* disp_map, int height, int width, int max_area, float *output);

#endif // LIBS

