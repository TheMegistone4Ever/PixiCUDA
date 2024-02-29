#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cualgo/negative.cuh"

#ifdef __INTELLISENSE__
#include "cualgo/intellisense/intellisense_cuda_intrinsics.hpp"
#endif

__global__ void negative_cuda(
	unsigned char* image,
	const int channels
);

void cuda_negative_image(
	unsigned char* image,
	const int height,
	const int width,
	const int channels
) {
	const int image_size = height * width * channels * sizeof(unsigned char);

	unsigned char* device_image = nullptr;

	// Allocate CUDA variable memory on the device
	cudaMalloc((void**)&device_image, image_size);

	// Copy the host variables to the device (CPU -> GPU)
	cudaMemcpy(device_image, image, image_size, cudaMemcpyHostToDevice);

	// Kernel launch
	dim3 grid_image(width, height);
	negative_cuda << <grid_image, 1 >> > (device_image, channels);

	// Copy the device variables to the host (GPU -> CPU)
	cudaMemcpy(image, device_image, image_size, cudaMemcpyDeviceToHost);

	// Free up the memory on the device (GPU)
	cudaFree(device_image);
}

__global__ void negative_cuda(
	unsigned char* image,
	const int channels
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = (x + y * gridDim.x) * channels;

	for (int i = 0; i < channels; i++) {
		image[index + i] = 255 - image[index + i];
	}
}

//__global__ void histogram_cuda(
//    const unsigned char* image,
//    const int width,
//    const int height,
//    const int channels,
//    int* histogram_r,
//    int* histogram_g,
//    int* histogram_b
//);
//
//void cuda_calculate_histogram(
//    const unsigned char* image,
//    const int height,
//    const int width,
//    const int channels,
//    int* histogram_r,
//    int* histogram_g,
//    int* histogram_b
//) {
//    const int image_size = height * width * channels * sizeof(unsigned char);
//    const int histogram_size = 256 * sizeof(int);
//
//    unsigned char* device_image = nullptr;
//    int* device_histogram_r = nullptr;
//    int* device_histogram_g = nullptr;
//    int* device_histogram_b = nullptr;
//
//    // Allocate CUDA variable memory on the device
//    cudaMalloc((void**)&device_image, image_size);
//    cudaMalloc((void**)&device_histogram_r, histogram_size);
//    cudaMalloc((void**)&device_histogram_g, histogram_size);
//    cudaMalloc((void**)&device_histogram_b, histogram_size);
//
//    // Copy the host variables to the device (CPU -> GPU)
//    cudaMemcpy(device_image, image, image_size, cudaMemcpyHostToDevice);
//    cudaMemcpy(device_histogram_r, histogram_r, histogram_size, cudaMemcpyHostToDevice);
//    cudaMemcpy(device_histogram_g, histogram_g, histogram_size, cudaMemcpyHostToDevice);
//    cudaMemcpy(device_histogram_b, histogram_b, histogram_size, cudaMemcpyHostToDevice);
//
//    // Kernel launch
//    dim3 grid_image(width, height);
//    dim3 block_image(1, 1);
//    histogram_cuda << <grid_image, block_image >> > (device_image, width, height, channels, device_histogram_r, device_histogram_g, device_histogram_b);
//
//    // Copy the device variables to the host (GPU -> CPU)
//    cudaMemcpy(histogram_r, device_histogram_r, histogram_size, cudaMemcpyDeviceToHost);
//    cudaMemcpy(histogram_g, device_histogram_g, histogram_size, cudaMemcpyDeviceToHost);
//    cudaMemcpy(histogram_b, device_histogram_b, histogram_size, cudaMemcpyDeviceToHost);
//
//    // Free up the memory on the device (GPU)
//    cudaFree(device_image);
//    cudaFree(device_histogram_r);
//    cudaFree(device_histogram_g);
//    cudaFree(device_histogram_b);
//}
//
//__global__ void histogram_cuda(
//    const unsigned char* image,
//    const int width,
//    const int height,
//    const int channels,
//    int* histogram_r,
//    int* histogram_g,
//    int* histogram_b
//) {
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//    int index = (y * gridDim.x + x) * channels;
//
//    atomicAdd(&histogram_r[image[index + 0]], 1);
//    atomicAdd(&histogram_g[image[index + 1]], 1);
//    atomicAdd(&histogram_b[image[index + 2]], 1);
//}
