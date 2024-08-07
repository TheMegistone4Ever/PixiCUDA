﻿#ifdef __INTELLISENSE__
    #define CUDA_KERNEL(...)
    #define FAKEINIT = { 0 }
    #define __CUDACC__
#else
    #define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
    #define FAKEINIT
#endif // __INTELLISENSE__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cualgo/motion_blur.cuh"
#include "utils/constants.hpp"

__constant__ unsigned char device_kernel[BIT_VECTOR_SIZE] FAKEINIT;

__global__ void motion_blur_cuda(
    const unsigned char* in_image,
    unsigned char* out_image,
    const int kernel_size,
    const int ones,
    const int height,
    const int width,
    const int channels
);

__host__ void set_bit_cuda_host(
    unsigned char* bit_vec,
    const int index
);

__device__ bool test_bit_cuda_device(
    const unsigned char* bit_vec,
    const int index
);

__host__ bool test_bit_cuda_host(
    const unsigned char* bit_vec,
    const int index
);

int create_kernel_cuda(
    const float angle_deg,
    const int distance,
    const int kernel_size,
    unsigned char* kernel
);

void cuda_motion_blur_image(
    const unsigned char* in_image,
    unsigned char* out_image,
    const float angle_deg,
    const int distance,
    const int height,
    const int width,
    const int channels,
    const unsigned int number_of_threads
)
{
    const size_t image_size = static_cast<size_t>(height) * width * channels * sizeof(unsigned char);

    // Create the kernel and fill it with the correct values
    int kernel_size = distance * 2 + 1; // +1 to include the center pixel
    unsigned char* host_kernel = new unsigned char[BIT_VECTOR_SIZE] { 0 };
    int ones = create_kernel_cuda(angle_deg, distance, kernel_size, host_kernel);

    // Copy the host kernel data to the device constant memory
    cudaMemcpyToSymbol(device_kernel, host_kernel, BIT_VECTOR_SIZE);

    // Allocate CUDA variable memory on the device
    unsigned char* device_in_image = nullptr;
    unsigned char* device_out_image = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&device_in_image), image_size);
    cudaMalloc(reinterpret_cast<void**>(&device_out_image), image_size);
    cudaMemset(device_out_image, NULL, image_size);

    // Copy the host variables to the device (CPU -> GPU)
    cudaMemcpy(device_in_image, in_image, image_size, cudaMemcpyHostToDevice);

    // Kernel launch
    dim3 block(number_of_threads);
    dim3 grid((width + block.x - 1) / block.x, height);

    motion_blur_cuda CUDA_KERNEL(grid, block)
    (
        device_in_image,
        device_out_image,
        kernel_size,
        ones,
        height,
        width,
        channels
    );

    // Copy the device variables to the host (GPU -> CPU)
    cudaMemcpy(out_image, device_out_image, image_size, cudaMemcpyDeviceToHost);

    // Free up the memory on the device (GPU)
    cudaFree(device_in_image);
    cudaFree(device_out_image);

    // Free up the memory on the host (CPU)
    delete[] host_kernel;
}

int create_kernel_cuda(
    const float angle_deg,
    const int distance,
    const int kernel_size,
    unsigned char* kernel
)
{
    float angle_rad = angle_deg * DEG_TO_RAD;
    int ones = 0;

    for (int i = 0; i < kernel_size; ++i)
    {
        int x = distance + int(i * cos(angle_rad));
        int y = distance + int(i * sin(angle_rad));
        int index = y * kernel_size + x;

        if (x < 0 || x >= kernel_size || y < 0 || y >= kernel_size)
        {
            break;
        }

        if (!test_bit_cuda_host(kernel, index))
        {
            set_bit_cuda_host(kernel, index);
            ++ones;
        }
    }

    return ones;
}

__global__ void motion_blur_cuda(
    const unsigned char* in_image,
    unsigned char* out_image,
    const int kernel_size,
    const int ones,
    const int height,
    const int width,
    const int channels
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
	{
		return;
	}

    int start_kernel_x = x - kernel_size / 2;
    int start_kernel_y = y - kernel_size / 2;
    int index = (x + y * width) * channels;
    int double_width = 2 * width;
    int double_height = 2 * height;

    for (int channel = 0; channel < channels; ++channel)
    {
        unsigned int channel_sum = 0;

        for (int x_kernel = start_kernel_x; x_kernel < start_kernel_x + kernel_size; ++x_kernel)
        {
            int reflected_x = abs(x_kernel) % double_width;
            if (reflected_x >= width)
            {
                reflected_x = double_width - reflected_x - 1;
            }

            for (int y_kernel = start_kernel_y; y_kernel < start_kernel_y + kernel_size; ++y_kernel)
            {
                int reflected_y = abs(y_kernel) % double_height;
                if (reflected_y >= height)
                {
                    reflected_y = double_height - reflected_y - 1;
                }

                unsigned char pixel = in_image[(reflected_x + reflected_y * width) * channels + channel];

                int kernel_index = (x_kernel - start_kernel_x) + (y_kernel - start_kernel_y) * kernel_size;
                unsigned char kernel_value = test_bit_cuda_device(device_kernel, kernel_index);

                if (kernel_value)
                {
                    channel_sum += pixel;
                }
            }
        }

        out_image[index + channel] = channel_sum / ones;
    }
}

__host__ void set_bit_cuda_host(
    unsigned char* bit_vec,
    const int index
)
{
    bit_vec[index / BYTE_SIZE] |= 1 << (index % BYTE_SIZE);
}

__device__ bool test_bit_cuda_device(
    const unsigned char* bit_vec,
    const int index
)
{
    return bit_vec[index / BYTE_SIZE] & (1 << (index % BYTE_SIZE));
}

__host__ bool test_bit_cuda_host(
    const unsigned char* bit_vec,
    const int index
)
{
    return bit_vec[index / BYTE_SIZE] & (1 << (index % BYTE_SIZE));
}
