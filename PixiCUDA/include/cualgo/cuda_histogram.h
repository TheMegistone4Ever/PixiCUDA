#pragma once


#ifndef _CUDA_HISTOGRAM_H_
#define _CUDA_HISTOGRAM_H_

void cuda_calculate_histogram(
	const unsigned char* image,
	const int height,
	const int width,
	const int channels,
	int* histogram_r,
	int* histogram_g,
	int* histogram_b
);

#endif // !_CUDA_HISTOGRAM_H_