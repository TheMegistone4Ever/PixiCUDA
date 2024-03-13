#pragma once

#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

constexpr unsigned short MAX_KERNEL_SIZE = 724;
constexpr unsigned char BYTE_SIZE = 8;
constexpr float DEG_TO_RAD = .017453293f;
constexpr unsigned char ESC_KEY = 27;
constexpr unsigned char Q_KEY = 113;
constexpr unsigned char WAIT_TIME = 50;
constexpr unsigned short WIN_WIDTH = 1280;
constexpr unsigned short WIN_HEIGHT = 720;
constexpr unsigned short MAX_ANGLE_DEG = 360;

#define BIT_VECTOR_SIZE (MAX_KERNEL_SIZE * MAX_KERNEL_SIZE / BYTE_SIZE + (MAX_KERNEL_SIZE * MAX_KERNEL_SIZE % BYTE_SIZE != 0))
#define MAX_DISTANCE (MAX_KERNEL_SIZE / 2 - 1)

#endif // !_CONSTANTS_H_
