#include <chrono>
#include <iomanip>
#include <iostream>
#include <filesystem>
//#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cualgo/motion_blur.cuh"
#include "cpualgo/motion_blur.hpp"
#include "utils/constants.hpp"
#include "utils/mae.hpp"

using namespace std;
using namespace cv;
namespace fs = filesystem;

typedef chrono::high_resolution_clock::time_point timevar;

static void onTrackbarAngle(int angle, void* userdata)
{
    *(float*)userdata = static_cast<float>(angle);
}

static void onTrackbarDistance(int distance, void* userdata)
{
    *(unsigned int*)userdata = static_cast<unsigned int>(distance);
}

static void onTrackbarAlgoSelection(int selection, void* userdata)
{
    *(unsigned int*)userdata = static_cast<unsigned int>(selection);
}

static void onCheckPrecision(int state, void* userdata)
{
    *(bool*)userdata = static_cast<bool>(state);
}

int main(int argc, char** argv)
{
    string images_path = R"(C:\Users\megis\VSProjects\PixiCUDA\PixiCUDA\images)";
    string image_name = "ny_ts_cropp_macdonalds.png";

    fs::path image_path = fs::path(images_path) / image_name;

    cout << "Image path: " << image_path << "\n";
    if (!fs::exists(image_path))
    {
        cout << "Could not open or find the image..." << endl;
        return ERROR_CODE;
    }
    Mat image = imread(image_path.string());

    // Image properties
    cout << "\t- Image height: " << image.rows << "\n"
        << "\t- Image width: " << image.cols << "\n"
        << "\t- Image channels: " << image.channels() << "\n";

    // This is the main window where all the magic happens
    namedWindow("Motion Blur", WINDOW_NORMAL);
    resizeWindow("Motion Blur", WIN_WIDTH, WIN_HEIGHT);

    float angle_deg = MIN_ANGLE_DEG;
    unsigned int distance = MIN_DISTANCE;
    unsigned int algo_selection = CPU;
    bool check_precision = PRECISION_OFF;

    float prev_angle_deg = angle_deg;
    unsigned int prev_distance = distance;
    unsigned int prev_algo_selection = algo_selection;
    bool prev_check_precision = check_precision;

    // Here the trackbars to control the motion blur parameters
    createTrackbar("Angle", "Motion Blur", MIN_ANGLE_DEG, MAX_ANGLE_DEG, onTrackbarAngle, &angle_deg);
    createTrackbar("Distance", "Motion Blur", MIN_DISTANCE, MAX_DISTANCE, onTrackbarDistance, &distance);
    createTrackbar("Algorithm", "Motion Blur", CPU, CUDA, onTrackbarAlgoSelection, &algo_selection);
    createTrackbar("Precision", "Motion Blur", PRECISION_OFF, PRECISION_ON, onCheckPrecision, &check_precision);

    Mat motion_blur_image = image.clone();
    Mat motion_blur_image_temp = image.clone();
    Mat stacked_image;
    hconcat(image, motion_blur_image, stacked_image);
    
    while (true)
    {
        char key = waitKey(WAIT_TIME);
        if (key == ESC_KEY || key == Q_KEY)
        {
            break;
        }

        if (prev_angle_deg != angle_deg ||
            prev_distance != distance ||
            prev_algo_selection != algo_selection ||
            prev_check_precision != check_precision
        )
        {
            cout << "\nMotion blur parameters changed:\n"
                << "\t- Angle: " << angle_deg << " degrees...\n"
                << "\t- Distance: " << distance << " pixels...\n"
                << "\t- Algorithm: ";

            timevar start = chrono::high_resolution_clock::now();
            switch (algo_selection)
            {
                case CPU:
                    cpu_motion_blur_image(
						image.data,
						motion_blur_image.data,
						angle_deg,
						distance,
						image.rows,
						image.cols,
						image.channels()
					);
					cout << "CPU...\n";
					break;
                
                case CUDA:
                    cuda_motion_blur_image(
                        image.data,
                        motion_blur_image.data,
                        angle_deg,
                        distance,
                        image.rows,
                        image.cols,
                        image.channels()
                    );
                    cout << "CUDA...\n";
                    break;

                default:
                    cout << "Invalid selection...\n";
            }
            timevar end = chrono::high_resolution_clock::now();

            cout << "\t- Algorithm time: "
                << chrono::duration_cast<chrono::milliseconds>(end - start).count()
                << " ms...\n";

            start = chrono::high_resolution_clock::now();
            hconcat(image, motion_blur_image, stacked_image);
            end = chrono::high_resolution_clock::now();

            cout << "\t- Stacking time: "
                << chrono::duration_cast<chrono::milliseconds>(end - start).count()
                << " ms...\n";

            if (check_precision)
            {
                cout << "\t- Mean Absolute Error (MAE): ";
                switch (algo_selection)
                {
                    case CPU:
                        cout << fixed << setprecision(32)
                            << .0
                            << defaultfloat << "%...\n";
                        break;

                    case CUDA:
                        cpu_motion_blur_image(
							image.data,
                            motion_blur_image_temp.data,
							angle_deg,
							distance,
							image.rows,
							image.cols,
							image.channels()
						);
						cout << fixed << setprecision(32)
                            << immae(motion_blur_image, motion_blur_image_temp) * 100
                            << defaultfloat << "%...\n";
                        break;

					default:
						cout << "Invalid selection...\n";
                }
			}

            prev_angle_deg = angle_deg;
            prev_distance = distance;
            prev_algo_selection = algo_selection;
            prev_check_precision = check_precision;
        }

        imshow("Motion Blur", stacked_image);
    }

    destroyAllWindows();

    imwrite("motion_blur_image.png", motion_blur_image);
    
    return SUCCESS_CODE;
}
