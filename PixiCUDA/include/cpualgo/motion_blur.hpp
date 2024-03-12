#pragma once


#ifndef _CPU_MOTION_BLUR_H_
#define _CPU_MOTION_BLUR_H_

constexpr unsigned short MAX_KERNEL_SIZE = 724;
constexpr float DEG_TO_RAD = .017453293;

void cpu_motion_blur_image(
	const unsigned char* in_image,
	unsigned char* out_image,
	const float angle_deg,
	const int distance,
	const int height,
	const int width,
	const int channels
);

#endif // !_CPU_MOTION_BLUR_H_
