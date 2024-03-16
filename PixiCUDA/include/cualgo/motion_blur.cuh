#pragma once

#ifndef _CUDA_MOTION_BLUR_H_
#define _CUDA_MOTION_BLUR_H_

void cuda_motion_blur_image(
	const unsigned char* in_image,
	unsigned char* out_image,
	const float angle_deg,
	const int distance,
	const int height,
	const int width,
	const int channels,
	const unsigned int number_of_threads
);

#endif // !_CUDA_MOTION_BLUR_H_
