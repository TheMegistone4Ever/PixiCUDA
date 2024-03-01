/**
* @file main.cpp
* * @brief Main file for the PixiCUDA project.
* * @date 2024-02-09
* * 
* * src/ (source for the application)
* * include/ (interface for the library *.h)
* * tests/ (main.cpp for quick tests) <- use cppunit for this part
* * doc/ (doxygen or any kind of documentation)
* *
* * include - PUBLIC header files (.h files).
* * src - PRIVATE source files (.h, .cpp and .m files).
* * test - tests files if you write tests (indefinitely you should).
* * doc - documentation files.
* */
#define DEG_TO_RAD 0.01745329252f
#define RED_BOLD "\033[1;31m"
#define GREEN_BOLD "\033[1;32m"
#define BLUE_BOLD "\033[1;34m"
#define RESET_COLOR "\033[0m"

#include <iostream>
#include <chrono>
#include <filesystem>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

//#include "cualgo/cuda_histogram.h"
//#include "cualgo/negative.cuh"
#include "cualgo/motion_blur.cuh"

using namespace std;
using namespace cv;
namespace fs = filesystem;

typedef chrono::high_resolution_clock::time_point TimeVar;

int main(int argc, char** argv)
{
	string images_path = R"(C:\Users\megis\VSProjects\PixiCUDA\PixiCUDA\images)";
	string image_name = R"(ny_ts_cropp_macdonalds.png)";

	fs::path image_path = fs::path(images_path) / image_name;

	//float angle_deg = 45;
	//int distance = 5;
	//
	//float* kernel = new float[(distance * 2 + 1) * (distance * 2 + 1)];
	//
	//int grid_kernel_size = fill_k(kernel, angle_deg, distance);
	//
	//Mat kernel_image = Mat(distance * 2 + 1, distance * 2 + 1, CV_32F, kernel);
	
	cout << "Image path: " << image_path << "\n";
	if (!fs::exists(image_path))
	{
		cout << "Could not open or find the image..." << endl;
		return -1;
	}
	Mat image = imread(image_path.string());

	// Image properties
	cout << "\tImage height: " << image.rows << "\n";
	cout << "\tImage width: " << image.cols << "\n";
	cout << "\tImage channels: " << image.channels() << "\n";

	// Motion Blur Image
	Mat motion_blur_image = image.clone();
	cuda_motion_blur_image(
		image.data,
		motion_blur_image.data,
		30.5,
		20,
		image.rows,
		image.cols,
		image.channels()
	);
	imwrite("motion_blur_image.png", motion_blur_image);
	imshow("Motion Blur Image", motion_blur_image);
	waitKey(0);
	destroyAllWindows();

	//cuda_negative_image(image.data, image.rows, image.cols, image.channels());
	//imwrite("negative_image.png", image);
	//int histogram_r[256] = { 0 };
	//int histogram_g[256] = { 0 };
	//int histogram_b[256] = { 0 };
	//
	//cuda_calculate_histogram(image.data, image.rows, image.cols, image.channels(), histogram_r, histogram_g, histogram_b);
	//
	//int total_pixels = image.rows * image.cols;
	//int total_pixels_histogram = 0;
	//
	//cout << "histogram: \n";
	//cout << "value\tR\tG\tB\n" << "—\t—\t—\t—\n";
	//for (int i = 0; i < sizeof(histogram_r) / sizeof(histogram_r[0]); i++)
	//{
	//	cout << i << ":";
	//	cout << "\t|" << RED_BOLD << histogram_r[i] << RESET_COLOR;
	//	cout << "\t|" << GREEN_BOLD << histogram_g[i] << RESET_COLOR;
	//	cout << "\t|" << BLUE_BOLD << histogram_b[i] << RESET_COLOR;
	//	cout << "\n";
	//	total_pixels_histogram += histogram_r[i] + histogram_g[i] + histogram_b[i];
	//}
	//total_pixels_histogram /= image.channels();
	//
	//cout << "Total pixels: " << total_pixels << "\n";
	//cout << "Total pixels histogram: " << total_pixels_histogram << "\n";
	//cout << "Remaining pixels: " << total_pixels - total_pixels_histogram << "\n";
	//
	////imshow("Original Image", image);
	////waitKey(0);
	////destroyAllWindows();

	return 0;
}
