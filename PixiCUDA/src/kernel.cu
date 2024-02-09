#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cualgo/cuda_histogram.h"

__global__ void histogram_cuda(
	const unsigned char* image,
	int* histogram_grayscale
);

void cuda_calculate_histogram(
	const unsigned char* image,
	const int height,
	const int width,
	const int channels,
	int* histogram_grayscale
) {
	unsigned char* device_image = nullptr;
	int* device_histogram = nullptr;

	const int image_size = height * width * channels * sizeof(int);
	const int histogram_size = 256 * sizeof(int);

	// Allocate CUDA variable memory on the device
	cudaMalloc((void**)&device_image, image_size);
	cudaMalloc((void**)&device_histogram, histogram_size);

	// Copy the host variables to the device (CPU -> GPU)
	cudaMemcpy(device_image, image, image_size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_histogram, histogram_grayscale, histogram_size, cudaMemcpyHostToDevice);

	// Kernel launch
	dim3 grid_image(width, height);
	dim3 block_dim(1, 1);
	histogram_cuda<<<grid_image, block_dim>>>(device_image, device_histogram);

	// Copy the device variables to the host (GPU -> CPU)
	cudaMemcpy(histogram_grayscale, device_histogram, histogram_size, cudaMemcpyDeviceToHost);

	// Free up the memory on the device (GPU)
	cudaFree(device_image);
	cudaFree(device_histogram);
}

__global__ void histogram_cuda(
	const unsigned char* image,
	int* histogram_grayscale
) {
	int x = blockIdx.x;
	int y = blockIdx.y;

	int index = y * gridDim.x + x;

	atomicAdd(&histogram_grayscale[image[index]], 1);
}