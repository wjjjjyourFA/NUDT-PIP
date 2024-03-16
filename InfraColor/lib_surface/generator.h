#ifndef FILTERLIDAR
#define FILTERLIDAR
#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda.h>

extern "C"
{
    void Generator(const float *xyz1, const float *xyz2, const float *face2, int height, int width, const float *k, const float *p, float *optical_flow, float *scene_flow);
}
__global__ void Generator_kernel(const float *xyz1, const float *xyz2, const float *face2, int height, int width, const float *k, const float *p, float *optical_flow, float *scene_flow);

#endif // FILTERLIDAR

