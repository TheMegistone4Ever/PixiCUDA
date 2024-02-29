#pragma once


#ifndef _CUDA_NEGATIVE_H_
#define _CUDA_NEGATIVE_H_

void cuda_negative_image(
	unsigned char* image,
	const int height,
	const int width,
	const int channels
);

#endif // !_CUDA_NEGATIVE_H_