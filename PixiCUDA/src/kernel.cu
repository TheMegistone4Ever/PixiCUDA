#include <cuda_runtime.h>
#include <crt/math_functions.hpp>
#include <device_launch_parameters.h>
#include "cualgo/motion_blur.cuh"

#include <stdio.h>

#ifdef __INTELLISENSE__
#include "cualgo/intellisense/intellisense_cuda_intrinsics.hpp"
#endif // __INTELLISENSE__

constexpr auto BYTE_SIZE = 8;
#define BIT_VECTOR_SIZE (MAX_KERNEL_SIZE * MAX_KERNEL_SIZE / BYTE_SIZE + (MAX_KERNEL_SIZE * MAX_KERNEL_SIZE % BYTE_SIZE != 0))

__constant__ unsigned char device_kernel[BIT_VECTOR_SIZE] = { 0 };

__host__ void setBit(unsigned char* bvec, const int index)
{
    bvec[index / BYTE_SIZE] |= 1 << (index % BYTE_SIZE);
}

__device__ bool testBit(const unsigned char* bvec, const int index)
{
    return bvec[index / BYTE_SIZE] & (1 << (index % BYTE_SIZE));
}

__global__ void motion_blur_cuda(
    const unsigned char* in_image,
    unsigned char* out_image,
    const int kernel_size,
    const int channels,
    const int height,
    const int width,
    const float angle_deg,
    const int distance
);

void cuda_motion_blur_image(
    const unsigned char* in_image,
    unsigned char* out_image,
    const float angle_deg,
    const int distance,
    const int height,
    const int width,
    const int channels
)
{
    // Calculate the size of the image
    const size_t image_size = height * width * channels * sizeof(unsigned char);

    // Create the kernel and fill it with the correct values
    float angle_rad = angle_deg * DEG_TO_RAD;
    int kernel_size = distance * 2 + 1; // +1 to include the center pixel
    unsigned char* host_kernel = new unsigned char[BIT_VECTOR_SIZE] { 0 };

    for (int i = 0; i < kernel_size; i++)
    {
        int x = distance + int(i * cos(angle_rad));
        int y = distance + int(i * sin(angle_rad));
        if (x >= 0 && x < kernel_size && y >= 0 && y < kernel_size)
        {
            setBit(host_kernel, y * kernel_size + x);
        }
    }

    // Copy the host kernel data to the device constant memory
    cudaMemcpyToSymbol(device_kernel, host_kernel, BIT_VECTOR_SIZE);

    // Allocate CUDA variable memory on the device
    unsigned char* device_in_image = nullptr;
    unsigned char* device_out_image = nullptr;

    cudaMalloc((void**)&device_in_image, image_size);
    cudaMalloc((void**)&device_out_image, image_size);
    cudaMemset(device_out_image, NULL, image_size);

    // Copy the host variables to the device (CPU -> GPU)
    cudaMemcpy(device_in_image, in_image, image_size, cudaMemcpyHostToDevice);

    // Kernel launch
    dim3 grid_image(width, height);
    motion_blur_cuda << <grid_image, 1 >> > (device_in_image, device_out_image, kernel_size, channels, height, width, angle_deg, distance);

    // Copy the device variables to the host (GPU -> CPU)
    cudaMemcpy(out_image, device_out_image, image_size, cudaMemcpyDeviceToHost);

    // Free up the memory on the device (GPU)
    cudaFree(device_in_image);
    cudaFree(device_out_image);
}

__global__ void motion_blur_cuda(
    const unsigned char* in_image,
    unsigned char* out_image,
    const int kernel_size,
    const int channels,
    const int height,
    const int width,
    const float angle_deg,
    const int distance
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = (x + y * width) * channels;

    if (x < width && y < height)
    {
        int start_kernel_x = x - kernel_size / 2;
        int start_kernel_y = y - kernel_size / 2;

        for (int channel = 0; channel < channels; channel++)
        {
            double channel_sum = 0;
            for (int x_kernel = start_kernel_x; x_kernel < start_kernel_x + kernel_size; x_kernel++)
            {
                if (x_kernel < 0 || x_kernel >= width)
                {
                    continue;
                }

                for (int y_kernel = start_kernel_y; y_kernel < start_kernel_y + kernel_size; y_kernel++)
                {
                    if (y_kernel < 0 || y_kernel >= height)
                    {
                        continue;
                    }

                    unsigned char pixel = in_image[(x_kernel + y_kernel * width) * channels + channel];

                    int kernel_index = (x_kernel - start_kernel_x) + (y_kernel - start_kernel_y) * kernel_size;
                    unsigned char kernel_value = testBit(device_kernel, kernel_index);

                    if (kernel_value)
                    {
                        channel_sum += static_cast<double>(pixel) / (distance + 1);
                    }
                }
            }
            out_image[index + channel] = static_cast<unsigned char>(channel_sum);
        }
    }
}
