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

#define MAX_DISTANCE (MAX_KERNEL_SIZE / 2 - 1)
constexpr unsigned short MAX_ANGLE_DEG = 360;
constexpr unsigned char ESC_KEY = 27;
constexpr unsigned char WAIT_TIME = 50;
constexpr unsigned short WIN_WIDTH = 1280;
constexpr unsigned short WIN_HEIGHT = 720;

#include <iostream>
#include <chrono>
#include <filesystem>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "cualgo/motion_blur.cuh"

using namespace std;
using namespace cv;
namespace fs = filesystem;

typedef chrono::high_resolution_clock::time_point TimeVar;

static void onTrackbarAngle(int angle, void* userdata)
{
    *(float*)userdata = static_cast<float>(angle);
}

static void onTrackbarDistance(int distance, void* userdata)
{
    *(unsigned int*)userdata = static_cast<unsigned int>(distance);
}

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
    Mat image = imread(image_path.string());

    // Image properties
    cout << "\tImage height: " << image.rows << "\n";
    cout << "\tImage width: " << image.cols << "\n";
    cout << "\tImage channels: " << image.channels() << "\n";

    float angle_deg;
    unsigned int distance;

    // Create a window with a stack of images
    namedWindow("Motion Blur", WINDOW_NORMAL);
    resizeWindow("Motion Blur", WIN_WIDTH, WIN_HEIGHT);

    // Create trackbars
    createTrackbar("Angle", "Motion Blur", 0, MAX_ANGLE_DEG, onTrackbarAngle, &angle_deg);
    createTrackbar("Distance", "Motion Blur", 0, MAX_DISTANCE, onTrackbarDistance, &distance);

    // Motion Blur Image
    Mat motion_blur_image = image.clone();
    Mat stacked_image;
    hconcat(image, motion_blur_image, stacked_image);
    imshow("Motion Blur", stacked_image);

    // Loop to handle events
    while (true)
    {
        cuda_motion_blur_image(
            image.data,
            motion_blur_image.data,
            angle_deg,
            distance,
            image.rows,
            image.cols,
            image.channels()
        );
        hconcat(image, motion_blur_image, stacked_image);
        imshow("Motion Blur", stacked_image);

        char key = waitKey(WAIT_TIME);
        if (key == ESC_KEY)
        {
            break;
        }    
    }

    destroyAllWindows();

    return 0;
}
