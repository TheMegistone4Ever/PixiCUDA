#include <filesystem>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cpualgo/motion_blur.hpp"
#include "cualgo/motion_blur.cuh"
#include "utils/constants.hpp"
#include "utils/mae.hpp"
#include "utils/main.hpp"
#include "utils/benchmark.hpp"

using namespace std;
using namespace cv;
namespace fs = filesystem;

typedef chrono::high_resolution_clock::time_point timevar;

int main(int argc, char** argv)
{
    return main_benchmark(5, 10, MAX_ANGLE_DEG, MAX_DISTANCE, 3,
        { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 }
    );

    string test_images_path = R"(images\test)";
    string image_name = "ny_ts_cropp_macdonalds.png";

    fs::path image_path = fs::path(test_images_path) / image_name;

    cout << BLUE_BOLD << "Image path: \"" << image_path << "\"\n";
    if (!fs::exists(image_path))
    {
        cerr << RED_BOLD << "Could not open or find the image;" << RESET_COLOR << endl;
        return ERROR_CODE;
    }
    Mat image = imread(image_path.string());

    // Image properties
    cout << CYAN_BOLD << "\t- Image height: " << image.rows
        << ";\n\t- Image width: " << image.cols
        << ";\n\t- Image channels: " << image.channels() << ";\n";

    // This is the main window where all the magic happens
    namedWindow(WIN_NAME, WINDOW_NORMAL);
    resizeWindow(WIN_NAME, WIN_WIDTH, WIN_HEIGHT);

    float angle_deg = MIN_ANGLE_DEG;
    unsigned int distance = MIN_DISTANCE;
    unsigned int algo_selection = CPU;
    bool check_precision = PRECISION_OFF;
    unsigned int threads_bin_log = MIN_THREADS_BLOG;

    float prev_angle_deg = angle_deg;
    unsigned int prev_distance = distance;
    unsigned int prev_algo_selection = algo_selection;
    bool prev_check_precision = check_precision;
    unsigned int prev_threads_bin_log = threads_bin_log;

    // Here the trackbars to control the motion blur parameters
    createTrackbar("Angle", WIN_NAME, MIN_ANGLE_DEG, MAX_ANGLE_DEG, onTrackbarAngle, &angle_deg);
    createTrackbar("Distance", WIN_NAME, MIN_DISTANCE, MAX_DISTANCE, onTrackbarDistance, &distance);
    createTrackbar("Algorithm", WIN_NAME, CPU, CUDA, onTrackbarAlgoSelection, &algo_selection);
    createTrackbar("Precision", WIN_NAME, PRECISION_OFF, PRECISION_ON, onTrackbarCheckPrecision, &check_precision);
    createTrackbar("Threads BL", WIN_NAME, MIN_THREADS_BLOG, MAX_THREADS_BLOG, onTrackbarThreadsBinLog, &threads_bin_log);

    Mat motion_blur_image = image.clone();
    Mat motion_blur_image_temp = image.clone();
    Mat stacked_image;
    unsigned int number_of_threads = 1 << threads_bin_log;
    long double mae = .0;
    long long cpu_mae_time = 0;
    hconcat(image, motion_blur_image, stacked_image);

    while (true)
    {
        char key = waitKey(WAIT_TIME);
        if (key == ESC_KEY || key == Q_KEY)
        {
            break;
        }

        if (prev_angle_deg != angle_deg
            || prev_distance != distance
            || prev_algo_selection != algo_selection
            || prev_check_precision != check_precision
            || (algo_selection != CPU && prev_threads_bin_log != threads_bin_log)
        )
        {
            cout << BLUE_BOLD << "\nMotion blur parameters changed:\n" << CYAN_BOLD
                << "\t- Angle: " << angle_deg << " degrees;\n"
                << "\t- Distance: " << distance << " pixels;\n"
                << "\t- Algorithm: ";

            timevar start = chrono::high_resolution_clock::now();
            switch (algo_selection)
            {
            case CPU:
                cout << "CPU;\n";
                cpu_motion_blur_image(
                    image.data,
                    motion_blur_image.data,
                    angle_deg,
                    distance,
                    image.rows,
                    image.cols,
                    image.channels()
                );
                break;

            case CUDA:
                number_of_threads = 1 << threads_bin_log;
                cout << "CUDA;\n\t\tL Number of threads: " << number_of_threads << ";\n";
                cuda_motion_blur_image(
                    image.data,
                    motion_blur_image.data,
                    angle_deg,
                    distance,
                    image.rows,
                    image.cols,
                    image.channels(),
                    number_of_threads
                );
                break;

            default:
                cerr << RED_BOLD << "Invalid selection;" << CYAN_BOLD << endl;
            }
            timevar end = chrono::high_resolution_clock::now();

            cout << "\t- Algorithm time: "
                << chrono::duration_cast<chrono::milliseconds>(end - start).count()
                << " ms;\n";

            start = chrono::high_resolution_clock::now();
            hconcat(image, motion_blur_image, stacked_image);
            end = chrono::high_resolution_clock::now();

            cout << "\t- Stacking time: "
                << chrono::duration_cast<chrono::milliseconds>(end - start).count()
                << " ms;\n";

            if (check_precision)
            {
                cout << "\t- Mean Absolute Error (MAE): ";
                switch (algo_selection)
                {
                case CPU:
                    cout << fixed << setprecision(32) << .0 << defaultfloat << "%;\n";
                    break;

                case CUDA:
                    start = chrono::high_resolution_clock::now();
                    cpu_motion_blur_image(
                        image.data,
                        motion_blur_image_temp.data,
                        angle_deg,
                        distance,
                        image.rows,
                        image.cols,
                        image.channels()
                    );
                    end = chrono::high_resolution_clock::now();
                    cpu_mae_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();

                    start = chrono::high_resolution_clock::now();
                    mae = immae(motion_blur_image, motion_blur_image_temp);
                    end = chrono::high_resolution_clock::now();

                    cout << fixed << setprecision(32) << mae * PERCENTAGE << defaultfloat
                        << "%;\n\t\tL CPU (MAE) time: " << cpu_mae_time
                        << " ms;\n\t\tL MAE time: "
                        << chrono::duration_cast<chrono::milliseconds>(end - start).count()
						<< " ms;\n";
                    break;

                default:
                    cerr << RED_BOLD << "Invalid selection;" << CYAN_BOLD << endl;
                }
            }

            prev_angle_deg = angle_deg;
            prev_distance = distance;
            prev_algo_selection = algo_selection;
            prev_check_precision = check_precision;
            prev_threads_bin_log = threads_bin_log;
        }

        imshow(WIN_NAME, stacked_image);
    }

    destroyAllWindows();

    imwrite(IMG_SAVE_PATH, motion_blur_image);

    cout << LIME_BOLD << "\nMotion blur image saved as \"" << IMG_SAVE_PATH
        << "\"...\nThanks for using this program!\n" << RESET_COLOR;

    return SUCCESS_CODE;
}

static void onTrackbarAngle(const int angle, void* userdata)
{
    *(float*)userdata = static_cast<float>(angle);
}

static void onTrackbarDistance(const int distance, void* userdata)
{
    *(unsigned int*)userdata = static_cast<unsigned int>(distance);
}

static void onTrackbarAlgoSelection(const int selection, void* userdata)
{
    *(unsigned int*)userdata = static_cast<unsigned int>(selection);
}

static void onTrackbarCheckPrecision(const int state, void* userdata)
{
    *(bool*)userdata = static_cast<bool>(state);
}

static void onTrackbarThreadsBinLog(const int threads, void* userdata)
{
    *(unsigned int*)userdata = static_cast<unsigned int>(threads);
}
