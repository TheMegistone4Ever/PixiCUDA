#pragma once

#ifndef _BENCHMARK_H_
#define _BENCHMARK_H_

#include <opencv2/core/mat.hpp>

int main_benchmark(
	const unsigned char warn_up,
	const unsigned char iterations,
	const float angle_deg,
	const unsigned int distance,
	const int channels,
	const std::vector<unsigned int> image_sizes
);
void generate_image(const int height, const int width, int channels, cv::Mat& image);

#endif // !_BENCHMARK_H_
