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

#include <iostream>
#include <chrono>
#include <filesystem>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "cualgo/cuda_histogram.h"

using namespace std;
using namespace cv;
namespace fs = filesystem;

typedef chrono::high_resolution_clock::time_point TimeVar;


int main(int argc, char** argv)
{
	string images_path = R"(C:\Users\megis\VSProjects\PixiCUDA\PixiCUDA\images)";
	string image_name = R"(ny_ts_cropp_macdonalds.png)";

	fs::path image_path = fs::path(images_path) / image_name;
	
	cout << "Image path: " << image_path << "\n";
	if (!fs::exists(image_path))
	{
		cout << "Could not open or find the image..." << endl;
		return -1;
	}
	Mat image = imread(image_path.string(), IMREAD_GRAYSCALE);

	// Image properties
	cout << "\tImage height: " << image.rows << "\n";
	cout << "\tImage width: " << image.cols << "\n";
	cout << "\tImage channels: " << image.channels() << "\n";

	int histogram_grayscale[256] = { 0 };

	cuda_calculate_histogram(image.data, image.rows, image.cols, image.channels(), histogram_grayscale);

	imshow("Original Image", image);
	
	cout << "histogram_grayscale: \n";
	for (int i = 0; i < sizeof(histogram_grayscale) / sizeof(histogram_grayscale[0]); i++)
	{
		cout << "\thistogram_grayscale[" << i << "]: " << histogram_grayscale[i] << "\n";
	}

	return 0;
}
