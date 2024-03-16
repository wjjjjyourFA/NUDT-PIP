#include "Getxyz.h"
#include "device_launch_parameters.h"

// 深度值的空白补全 取最近的深度
void GetXYZ(const float *disp_map, int height, int width, int max_area, float *output)
{
    float *disp_map_gpu;
    float *output_gpu;
//    在device上申请一定字节大小的显存，其中devPtr是指向所分配内存的指针。
    cudaMalloc(&disp_map_gpu, height * width * 3 * sizeof(float));
    cudaMalloc(&output_gpu, height * width * 4 * sizeof(float));
//    负责host和device之间数据通信 cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
    cudaMemcpy(disp_map_gpu, disp_map, height * width * 3 * sizeof(float), cudaMemcpyHostToDevice);
//    block 中有几列几行个 thread
    dim3 block(52, 16);
    dim3 gridSize(width / block.x + (width % block.x == 0 ? 0 : 1),
                  height / block.y + (height % block.y == 0 ? 0 : 1));
    cudaThreadSynchronize();
    GetXYZ_kernel<<<gridSize, block>>>(disp_map_gpu, height, width, max_area, output_gpu);
    cudaThreadSynchronize();
    cudaMemcpy(output, output_gpu, height * width * 4 * sizeof(float), cudaMemcpyDeviceToHost);
}

__global__ void GetXYZ_kernel(const float* disp_map, int height, int width, int max_area, float *output)
{
//    获取一个线程在blcok中的全局ID
    int x_id = blockIdx.x * blockDim.x + threadIdx.x;
    int y_id = blockIdx.y * blockDim.y + threadIdx.y;
    if(x_id >= width || y_id >= height)
        return;
//    printf("x_id: %d, y_id: %d\n", x_id, y_id);
    int idx = y_id * width + x_id;
    int map_size = width * height;

    output[idx] = 0;
    output[idx + map_size] = 0;
    output[idx + map_size * 2] = 0;
    output[idx + map_size * 3] = 0;
    if(disp_map[idx + map_size * 2] != 0.)
    {
        //printf("disp_map: %f, %f, %f\n", disp_map[idx], disp_map[idx + map_size], disp_map[idx + map_size * 2]);
        output[idx] = disp_map[idx];
        output[idx + map_size] = disp_map[idx + map_size];
        output[idx + map_size * 2] = disp_map[idx + map_size * 2];
        output[idx + map_size * 3] = 1;
        return;
    }
    float x = 0., y = 0., z = 0., flag = 0.;
    float minz = 1000.;
    int loop = 1;
    while(z == 0.)
    {
        if(loop > max_area)
            break;
        int start_row = MMAX(y_id - loop, 0);
        int end_row = MMIN(y_id + loop + 1, height - 1);
        int start_col = MMAX(x_id - loop, 0);
        int end_col = MMIN(x_id + loop + 1, width - 1);

        for(int j = start_col; j < end_col; j++)
        {
            uint32_t idx1 = start_row * width + j;
            float val1 = disp_map[idx1 + map_size * 2];
            if(val1 != 0.)
            {
                if(val1 < minz)
                {
                    x = disp_map[idx1];
                    y = disp_map[idx1 + map_size];
                    z = disp_map[idx1 + map_size * 2];
                    flag = 1;
                    minz = val1;
                }
            }

            uint32_t idx2 = end_row * width + j;
            float val2 = disp_map[idx2 + map_size * 2];
            if(val2 != 0.)
            {
                if(val2 < minz)
                {
                    x = disp_map[idx2];
                    y = disp_map[idx2 + map_size];
                    z = disp_map[idx2 + map_size * 2];
                    flag = 1;
                    minz = val2;
                }
            }
        }

        for(int i = start_row; i < end_row; i++)
        {
            uint32_t idx1 = i * width + start_col;
            float val1 = disp_map[idx1 + map_size * 2];
            if(val1 != 0.)
            {
                if(val1 < minz)
                {
                    x = disp_map[idx1];
                    y = disp_map[idx1 + map_size];
                    z = disp_map[idx1 + map_size * 2];
                    flag = 1;
                    minz = val1;
                }
            }

            uint32_t idx2 = i * width + end_col;
            float val2 = disp_map[idx2 + map_size * 2];
            if(val2 != 0.)
            {
                if(val2 < minz)
                {
                    x = disp_map[idx2];
                    y = disp_map[idx2 + map_size];
                    z = disp_map[idx2 + map_size * 2];
                    flag = 1;
                    minz = val2;
                }
            }
        }
        loop += 1;
    }
    output[idx] = x;
    output[idx + map_size] = y;
    output[idx + map_size * 2] = z;
    output[idx + map_size * 3] = flag;
}
