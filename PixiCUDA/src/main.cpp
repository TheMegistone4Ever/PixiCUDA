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

#define MAX_DISTANCE (MAX_KERNEL_SIZE / 2 - 1)
constexpr unsigned char ESC_KEY = 27;
constexpr unsigned char Q_KEY = 113;
constexpr unsigned char WAIT_TIME = 50;
constexpr unsigned short WIN_WIDTH = 1280;
constexpr unsigned short WIN_HEIGHT = 720;
constexpr unsigned short MAX_ANGLE_DEG = 360;

typedef chrono::high_resolution_clock::time_point timevar;

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
    string image_name = "ny_times_square.png";

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

    float angle_deg = .0f;
    unsigned int distance = 0;

    // Create a window with a stack of images
    namedWindow("Motion Blur", WINDOW_NORMAL);
    resizeWindow("Motion Blur", WIN_WIDTH, WIN_HEIGHT);

    // Create trackbars
    createTrackbar("Angle", "Motion Blur", 0, MAX_ANGLE_DEG, onTrackbarAngle, &angle_deg);
    createTrackbar("Distance", "Motion Blur", 0, MAX_DISTANCE, onTrackbarDistance, &distance);

    Mat motion_blur_image = image.clone();
    Mat stacked_image;

    float prev_angle_deg = angle_deg;
    unsigned int prev_distance = distance;

    hconcat(image, motion_blur_image, stacked_image);
    while (true)
    {
        char key = waitKey(WAIT_TIME);
        if (key == ESC_KEY || key == Q_KEY)
        {
            break;
        }

        // Check if parameters have changed
        if (prev_angle_deg != angle_deg || prev_distance != distance)
        {
            timevar start = chrono::high_resolution_clock::now();
            cuda_motion_blur_image(
                image.data,
                motion_blur_image.data,
                angle_deg,
                distance,
                image.rows,
                image.cols,
                image.channels()
            );
            timevar end = chrono::high_resolution_clock::now();
            cout << "CUDA time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms...\n";

            hconcat(image, motion_blur_image, stacked_image);

            prev_angle_deg = angle_deg;
            prev_distance = distance;
        }
        
        imshow("Motion Blur", stacked_image);
    }

    destroyAllWindows();

    imwrite("motion_blur_image.png", motion_blur_image);

    return 0;
}
