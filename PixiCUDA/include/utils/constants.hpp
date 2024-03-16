#pragma once

#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

constexpr char SUCCESS_CODE = 0;
constexpr char ERROR_CODE = -1;

constexpr unsigned short MAX_KERNEL_SIZE = 724;
constexpr unsigned char PERCENTAGE = 100;
constexpr unsigned char BYTE_SIZE = 8;
constexpr float DEG_TO_RAD = .017453293f;

constexpr unsigned char WAIT_TIME = 50;
constexpr unsigned short WIN_WIDTH = 1440;
constexpr unsigned short WIN_HEIGHT = 900;

constexpr unsigned char ESC_KEY = 27;
constexpr unsigned char Q_KEY = 113;

constexpr const char* WIN_NAME = "Motion Blur Filter (ESC/Q to exit)";
constexpr const char* IMG_SAVE_PATH = "motion_blur_image.png";

constexpr const char* CYAN_BOLD = "\033[1;96m";
constexpr const char* LIME_BOLD = "\033[1;92m";
constexpr const char* BLUE_BOLD = "\033[1;34m";
constexpr const char* RED_BOLD = "\033[1;91m";
constexpr const char* RESET_COLOR = "\033[0m";

constexpr unsigned short BIT_VECTOR_SIZE = MAX_KERNEL_SIZE * MAX_KERNEL_SIZE / BYTE_SIZE + (MAX_KERNEL_SIZE * MAX_KERNEL_SIZE % BYTE_SIZE != 0);

#define MIN_ANGLE_DEG 0
#define MAX_ANGLE_DEG 360

#define MIN_DISTANCE 0
#define MAX_DISTANCE (MAX_KERNEL_SIZE / 2 - 1)

#define MIN_THREADS_BLOG 0
#define MAX_THREADS_BLOG 10

#define CPU 0
#define CUDA 1

#define PRECISION_OFF 0
#define PRECISION_ON 1

#endif // !_CONSTANTS_H_
