#include <cmath>

#include "cpualgo/motion_blur.hpp"
#include "utils/constants.hpp"
#include <iostream>

unsigned char kernel[BIT_VECTOR_SIZE] = { 0 };

void motion_blur_cpu(
	const unsigned char* in_image,
	unsigned char* out_image,
	const int kernel_size,
	const float angle_deg,
	const int distance,
	const int height,
	const int width,
	const int channels
);

void setBitCpu(
	unsigned char* bit_vec,
	const int index
);

bool testBitCpu(
	const unsigned char* bit_vec,
	const int index
);

void cpu_motion_blur_image(
	const unsigned char* in_image,
	unsigned char* out_image,
	const float angle_deg,
	const int distance,
	const int height,
	const int width,
	const int channels
)
{
	// Create the kernel and fill it with the correct values
	float angle_rad = angle_deg * DEG_TO_RAD;
	int kernel_size = distance * 2 + 1; // +1 to include the center pixel

	for (int i = 0; i < kernel_size; i++)
	{
		int x = distance + int(i * cos(angle_rad));
		int y = distance + int(i * sin(angle_rad));

		if (x >= 0 && x < kernel_size && y >= 0 && y < kernel_size)
		{
			setBitCpu(kernel, y * kernel_size + x);
		}
		else
		{
			break;
		}
	}

	// Print kernel in nice format
	for (int i = 0; i < kernel_size; i++)
	{
		for (int j = 0; j < kernel_size; j++)
		{
			std::cout << testBitCpu(kernel, i * kernel_size + j) << " ";
		}
		std::cout << std::endl;
	}

	motion_blur_cpu(
		in_image,
		out_image,
		kernel_size,
		angle_deg,
		distance,
		height,
		width,
		channels
	);
}

void motion_blur_cpu(
	const unsigned char* in_image,
	unsigned char* out_image,
	const int kernel_size,
	const float angle_deg,
	const int distance,
	const int height,
	const int width,
	const int channels
)
{
	const size_t image_size = height * width * channels * sizeof(unsigned char);
	memset(out_image, NULL, image_size);

	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			int start_kernel_x = x - kernel_size / 2;
			int start_kernel_y = y - kernel_size / 2;
			int index = (x + y * width) * channels;

			for (int channel = 0; channel < channels; channel++)
			{
				unsigned int channel_sum = 0;

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
						unsigned char kernel_value = testBitCpu(kernel, kernel_index);

						if (kernel_value)
						{
							channel_sum += pixel;
						}
					}
				}

				out_image[index + channel] = channel_sum / (distance + 1.);
			}
		}
	}
}

void setBitCpu(
	unsigned char* bit_vec,
	const int index
)
{
	bit_vec[index / BYTE_SIZE] |= 1 << (index % BYTE_SIZE);
}

bool testBitCpu(
	const unsigned char* bit_vec,
	const int index
)
{
	return bit_vec[index / BYTE_SIZE] & (1 << (index % BYTE_SIZE));
}
