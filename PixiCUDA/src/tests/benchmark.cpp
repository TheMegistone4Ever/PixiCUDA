#include <opencv2/core/core.hpp>
#include <iostream>
#include <format>
#include <chrono>

#include "cualgo/motion_blur.cuh"
#include "cpualgo/motion_blur.hpp"
#include "tests/benchmark.hpp"
#include "utils/constants.hpp"
#include "utils/mae.hpp"

using namespace std;
using namespace cv;

typedef chrono::high_resolution_clock::time_point timevar;

int main_benchmark(
    const unsigned char warn_up,
    const unsigned char iterations,
    const float angle_deg,
    const unsigned int distance,
    const int channels,
    const vector<unsigned int> sizes
)
{
    vector<Mat> images(sizes.size());
    for (int i = 0; i < sizes.size(); ++i)
    {
        generate_image(sizes[i], sizes[i], channels, images[i]);
    }

    vector<unsigned int> threads(MAX_THREADS_BLOG - MIN_THREADS_BLOG + 1);
    for (int i = 0; i < threads.size(); ++i)
    {
        threads[i] = pow(2, i);
    }

    cout << std::format("Algorithm \"CPU\", warn up: {}, iterations: {}.\n", warn_up, iterations);
    cout << std::format("{:<16}{:<16}{:<35}{:<16}\n", "Image size", "Time (ms)", "MAE (%, CPU vs CPU)", "MAE time (ms)");
    vector<Mat> cpus(sizes.size());
    for (int s = 0; s < sizes.size(); ++s) {
        cpus[s] = images[s].clone();
        cout << std::format("{:<16}", to_string(sizes[s]) + "x" + to_string(sizes[s]));
        
        for (int i = 0; i < warn_up; ++i)
        {
            cpu_motion_blur_image(images[s].data, cpus[s].data, angle_deg, distance, sizes[s], sizes[s], channels);
        }

        timevar start = chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i)
        {
            cpu_motion_blur_image(images[s].data, cpus[s].data, angle_deg, distance, sizes[s], sizes[s], channels);
        }
        timevar end = chrono::high_resolution_clock::now();
        
        cout << std::format("{:<16}", chrono::duration_cast<chrono::milliseconds>(end - start).count() / iterations);

        start = chrono::high_resolution_clock::now();
        long double mae = immae(cpus[s], cpus[s]);
        end = chrono::high_resolution_clock::now();

        cout << fixed << setprecision(32) << mae * PERCENTAGE << defaultfloat;
        cout << std::format(" {:<16}\n", chrono::duration_cast<chrono::milliseconds>(end - start).count());
    }

    cout << std::format("\nAlgorithm \"CUDA\", warn up: {}, iterations: {}.\n{:<16}", warn_up, iterations, "Image size");
    for (const unsigned int thread : threads) {
		cout << std::format("{:<16}", "[" + to_string(thread) + " Thread" + (thread > 1 ? "s" : "") + "]");
	}
    cout << std::format("{:<35}{:<16}\n", "MAE (%, CUDA vs CPU)", "MAE time (ms)");
    Mat cuda;
    for (int s = 0; s < sizes.size(); ++s) {
        cuda = images[s].clone();
        cout << std::format("{:<16}", to_string(sizes[s]) + "x" + to_string(sizes[s]));
        
        for (const unsigned int thread : threads) {
            for (int i = 0; i < warn_up; ++i)
            {
                cuda_motion_blur_image(images[s].data, cuda.data, angle_deg, distance, sizes[s], sizes[s], channels, thread);
            }

            timevar start = chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i)
            {
                cuda_motion_blur_image(images[s].data, cuda.data, angle_deg, distance, sizes[s], sizes[s], channels, thread);
            }
            timevar end = chrono::high_resolution_clock::now();
            
            cout << std::format("{:<16}", chrono::duration_cast<chrono::milliseconds>(end - start).count() / iterations);
        }

        timevar start = chrono::high_resolution_clock::now();
        long double mae = immae(cuda, cpus[s]);
        timevar end = chrono::high_resolution_clock::now();

        cout << fixed << setprecision(32) << mae * PERCENTAGE << defaultfloat;
        cout << std::format(" {:<16}\n", chrono::duration_cast<chrono::milliseconds>(end - start).count());
    }

    return SUCCESS_CODE;
}

void generate_image(const int height, const int width, int channels, Mat& image)
{
    if (channels != 1 && channels != 3 && channels != 4)
	{
		channels = 1;
	}

    image = Mat::zeros(height, width, CV_8UC(channels));
	randu(image, Scalar::all(0), Scalar::all(UCHAR_MAX));
}
