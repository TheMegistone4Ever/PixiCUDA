#include <cuda_runtime.h>
#include <crt/math_functions.hpp>
#include <device_launch_parameters.h>
#include "cualgo/motion_blur.cuh"

#ifdef __INTELLISENSE__
#include "cualgo/intellisense/intellisense_cuda_intrinsics.hpp"
#endif

#define DEG_TO_RAD 0.01745329252f

// Function to apply motion blur to an image
__global__ void motion_blur_cuda(
	const unsigned char* in_image,
	unsigned char* out_image,
	const float* kernel,
	const int kernel_size,
	const int channels,
	const int height,
	const int width
);

void cuda_motion_blur_image(
	const unsigned char* in_image,
	unsigned char* out_image,
	const float angle_deg,
	const int distance,
	const int height,
	const int width,
	const int channels
) {
	// Calculate the size of the image
	const size_t image_size = height * width * channels * sizeof(unsigned char);
	
	// Create the kernel and fill it with the correct values

	float angle_rad = angle_deg * DEG_TO_RAD;

	int size = distance * 2 + 1; // +1 to include the center pixel
	int grid_kernel_size = size * size;
	size_t kernel_size = grid_kernel_size * sizeof(float);

	float* kernel = new float[grid_kernel_size];

	for (int i = 0; i < size; i++) {
		int x = distance + int(i * cos(angle_rad));
		int y = distance + int(i * sin(angle_rad));
		if (x < 0 || x >= size || y < 0 || y >= size) {
			break;
		}
		kernel[y * size + x] = 1. / (distance + 1);
	}

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
	motion_blur_cuda << <grid_image, 1 >> > (device_in_image, device_out_image, kernel, kernel_size, channels, height, width);

	// Copy the device variables to the host (GPU -> CPU)
	cudaMemcpy(out_image, device_out_image, image_size, cudaMemcpyDeviceToHost);

	// Free up the memory on the device (GPU)
	cudaFree(device_in_image);
	cudaFree(device_out_image);
}

__global__ void motion_blur_cuda(
	const unsigned char* in_image,
	unsigned char* out_image,
	const float* kernel,
	const int kernel_size,
	const int channels,
	const int height,
	const int width
) {
	// Get the position of the pixel
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = (x + y * gridDim.x) * channels;

	// Apply the motion blur to the pixel
	if (x < width && y < height) {
		for (int channel = 0; channel < channels; channel++) {
			float sum = 0;
			for (int i = 0; i < kernel_size; i++) {
				int x_kernel = i % kernel_size;
				int y_kernel = i / kernel_size;
				int x_image = x + x_kernel - kernel_size / 2;
				int y_image = y + y_kernel - kernel_size / 2;
				if (x_image >= 0 && x_image < width && y_image >= 0 && y_image < height) {
					atomicAdd(&sum, in_image[(x_image + y_image * width) * channels + channel] * kernel[i]);
				}
			}
			out_image[index + channel] = (int)sum;
		}
	}
}

//__global__ void negative_cuda(
//	unsigned char* image,
//	const int channels
//);
//
//void cuda_negative_image(
//	unsigned char* image,
//	const int height,
//	const int width,
//	const int channels
//) {
//	const int image_size = height * width * channels * sizeof(unsigned char);
//
//	unsigned char* device_image = nullptr;
//
//	// Allocate CUDA variable memory on the device
//	cudaMalloc((void**)&device_image, image_size);
//
//	// Copy the host variables to the device (CPU -> GPU)
//	cudaMemcpy(device_image, image, image_size, cudaMemcpyHostToDevice);
//
//	// Kernel launch
//	dim3 grid_image(width, height);
//	negative_cuda << <grid_image, 1 >> > (device_image, channels);
//
//	// Copy the device variables to the host (GPU -> CPU)
//	cudaMemcpy(image, device_image, image_size, cudaMemcpyDeviceToHost);
//
//	// Free up the memory on the device (GPU)
//	cudaFree(device_image);
//}
//
//__global__ void negative_cuda(
//	unsigned char* image,
//	const int channels
//) {
//	int x = blockIdx.x * blockDim.x + threadIdx.x;
//	int y = blockIdx.y * blockDim.y + threadIdx.y;
//	int index = (x + y * gridDim.x) * channels;
//
//	for (int i = 0; i < channels; i++) {
//		image[index + i] = 255 - image[index + i];
//	}
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
