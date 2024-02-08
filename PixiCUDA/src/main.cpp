#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <chrono>
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = filesystem;

typedef chrono::high_resolution_clock::time_point TimeVar;


int main(int argc, char** argv)
{
	string images_path = R"(C:\Users\megis\VSProjects\PixiCUDA\PixiCUDA\images)";
	string image_name = R"(ny_times_square.png)";

	fs::path image_path = fs::path(images_path) / image_name;
	
	cout << "Image path: " << image_path << "\n";
	if (!fs::exists(image_path))
	{
		cout << "Could not open or find the image..." << endl;
		return -1;
	}
	Mat image = imread(image_path.string());

	// Image properties
	cout << "Image height: " << image.rows << "\n";
	cout << "Image width: " << image.cols << "\n";
	cout << "Image channels: " << image.channels() << "\n";

	return 0;
}
