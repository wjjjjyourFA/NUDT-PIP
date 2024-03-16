#include "generator.h"
#include "device_launch_parameters.h"

// 图像和点云的变换
void Generator(const float *xyz1, const float *xyz2, const float *face2, int height, int width, const float *k, const float *p, float *optical_flow, float *scene_flow)
{
    float *xyz1_gpu;
    float *xyz2_gpu;
    float *face2_gpu;
    float *k_gpu;
    float *p_gpu;
    float *of_gpu;
    float *sf_gpu;

    cudaMalloc(&xyz1_gpu, height * width * 3 * sizeof(float));
    cudaMalloc(&k_gpu, 12 * sizeof(float));
    cudaMalloc(&xyz2_gpu, height * width * 3 * sizeof(float));
    cudaMalloc(&face2_gpu, height * width * 3 * sizeof(float));
    cudaMalloc(&p_gpu, 16 * sizeof(float));
    cudaMalloc(&of_gpu, height * width * 2 * sizeof(float));
    cudaMalloc(&sf_gpu, height * width * 3 * sizeof(float));

    cudaMemcpy(xyz1_gpu, xyz1, height * width * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(k_gpu, k, 12 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(xyz2_gpu, xyz2, height * width * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(face2_gpu, face2, height * width * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(p_gpu, p, 16 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(52, 16);
    dim3 gridSize(width / block.x + (width % block.x == 0 ? 0 : 1),
                  height / block.x + (height % block.x == 0 ? 0 : 1));

    cudaThreadSynchronize();
    Generator_kernel<<<gridSize, block>>>(xyz1_gpu, xyz2_gpu, face2_gpu, height, width, k_gpu, p_gpu, of_gpu, sf_gpu);
    cudaThreadSynchronize();
    cudaMemcpy(optical_flow, of_gpu, height * width * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(scene_flow, sf_gpu, height * width * 3 * sizeof(float), cudaMemcpyDeviceToHost);
}

__global__ void Generator_kernel(const float *xyz1, const float *xyz2, const float *face2, int height, int width, const float *k, const float *p, float *optical_flow, float *scene_flow)
{
    int x_id = blockIdx.x * blockDim.x + threadIdx.x;
    int y_id = blockIdx.y * blockDim.y + threadIdx.y;
    if(x_id >= width || y_id >= height)
        return;

    int idx = y_id * width + x_id;
    int map_size = width * height;
    float thread = 1.0;

    float x1 = xyz1[idx];
    float y1 = xyz1[idx + map_size];
    float z1 = xyz1[idx + map_size * 2];
//    printf("x_id: %d, y_id: %d, x1: %f, y1: %f, z1: %f\n", x_id, y_id, x1, y1, z1);
//    for(int i = 0; i < 16; i++)
//        printf("%f", p[i]);
//    printf("\n");

    float rot_x = p[0] * x1 + p[1] * y1 + p[2] * z1 + p[3];
    float rot_y = p[4] * x1 + p[5] * y1 + p[6] * z1 + p[7];
    float rot_z = p[8] * x1 + p[9] * y1 + p[10] * z1 + p[11];
//    printf("rot_x: %f, rot_y: %f, rot_z: %f\n", rot_x, rot_y, rot_z);

    float s = k[8] * rot_x + k[9] * rot_y + k[10] * rot_z + k[11];
    int u = (int)((k[0] * rot_x + k[1] * rot_y + k[2] * rot_z + k[3]) / s);
    int v = (int)((k[4] * rot_x + k[5] * rot_y + k[6] * rot_z + k[7]) / s);
//    printf("u: %f, v: %f, s: %f\n", u, v, s);

    if(v >= 0 && v < height && u >= 0 && u < width)
    {
//        printf("here2\n");
        int idx2 = v * width + u;
        float expected_z = face2[idx2 + map_size * 2];
        if(expected_z != 0 && expected_z - thread < rot_z && expected_z + thread > rot_z)
        {
            float x2 = xyz2[idx2];
            float y2 = xyz2[idx2 + map_size];
            float z2 = xyz2[idx2 + map_size * 2];
            optical_flow[idx] = (float)(u - x_id);
            optical_flow[idx + map_size] = (float)(v - y_id);
            printf("here3\n");
            if(x2 != 0 || y2 != 0 || z2 != 0)
            {
                printf("here4\n");
                scene_flow[idx] = x2 - x1;
                scene_flow[idx + map_size] = y2 - y1;
                scene_flow[idx + map_size * 2] = z2 - z1;
            }
        }
    }
}

