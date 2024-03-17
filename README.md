# PixiCUDA Motion Blur Filter for images with CUDA acceleration
###### &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; — by [Mykyta Kyselov (TheMegistone4Ever)](https://github.com/TheMegistone4Ever).

## Table of Contents

1. [Getting Started](#1-getting-started)
   1. [Deploying the Software](#11-deploying-the-software)
   2. [Purpose of Development](#12-purpose-of-development)
2. [Preparing to Work with the Software](#2-preparing-to-work-with-the-software)
   1. [System Requirements](#21-system-requirements)
   2. [Software Requirements](#22-software-requirements)
3. [Running the Program](#3-running-the-program)
   1. [The main page of the PixiCUDA](#31-the-main-page-of-the-pixicuda)
	  1. [An example of ...](#311-an-example-of-)
	  2. [An example of ...](#312-an-example-of-)
	  3. [An example of ...](#313-an-example-of-)
4. [License](#4-license)

## 1. Getting Started

## 1.1 Deploying the Software

To deploy the motion blur filter application, follow these steps:

- Install CUDA Toolkit 12.2 or later from the [official NVIDIA website](https://developer.nvidia.com/cuda-downloads).
- Install the latest version of [CMake](https://cmake.org/download/).
- Install the latest version of [Visual Studio](https://visualstudio.microsoft.com/).										
- Build the OpenCV library from source with CUDA 12.2 or later support for Visual Studio you have installed. The 
  instructions for building OpenCV with CUDA support can be
  found [here](https://docs.opencv.org/master/d6/d15/tutorial_building_tegra_cuda.html).
- Create a new `CUDA 12.2 (or or later) Runtime` project in Visual Studio or another C++ environment and clone the
  PixiCUDA repository:

  `git clone https://github.com/TheMegistone4Ever/PixiCUDA.git`

- Open the project in Visual Studio.
- Edit path to images in [main.cpp](PixiCUDA/src/main.cpp) file.
- Build the project in Visual Studio into an executable (.exe) file.
- Run the application.														
- Select needed parameters. Application will create a new image with motion blur effect.
- Enjoy the result!

## 1.2 Purpose of Development

The development aims to provide users with a simple and fast way to apply motion blur effect to images using CUDA
acceleration. The application is designed to be user-friendly and easy to use. The application is intended for users
who want to apply motion blur effect to images with CUDA acceleration. It also providef usual CPU version of motion
blur filter for time comparison.

## 2 Preparing to Work with the Software

### 2.1 System Requirements

**Minimum Hardware Configuration:**
- Processor type: [Intel Core i3](https://www.intel.com/content/www/us/en/products/details/processors/core/i3.html) or equivalent;
- RAM: 2 GB;
- Graphics card: [NVIDIA GeForce 600 series](https://www.nvidia.com/download/driverResults.aspx/218807/en-us) or later;
- Required Driver Version for [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html): >=525.60.13 for
  Linux, >=528.33 for Windows;
- Internet connection: Not required;
- Disk space: 1 GB;
- C++ compiler: [Visual Studio](https://visualstudio.microsoft.com) 2019 or later.

**Recommended Hardware Configuration:**
- Processor type: [AMD Ryzen 9 6900HS with Radeon Graphics](https://nanoreview.net/en/cpu/amd-ryzen-9-6900hs) or equivalent;
- RAM: 16 GB;
- Graphics card: [NVIDIA GeForce RTX 3050](https://www.nvidia.com/ru-ru/geforce/graphics-cards/30-series/rtx-3050) or later;
- Required Driver Version for [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html): >=525.60.13 for
  Linux, >=528.33 for Windows;
- Internet connection: Not required;
- Disk space: 2 GB;
- C++ compiler: [Visual Studio](https://visualstudio.microsoft.com) 2022 or later.

### 2.2 Software Requirements

**Minimum Software Configuration:**
- Operating system: [Windows 10](https://www.microsoft.com/en-us/windows/get-windows-10) or later;
- C++ compiler: [Visual Studio](https://visualstudio.microsoft.com) 2019 or later;
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 12.2 or later;
- [CMake](https://cmake.org/download/) 3.21 or later;
- [OpenCV](https://opencv.org/releases/) 4.5 or later.
- [Git](https://git-scm.com/downloads) 2.33 or later.

- **Recommended Software Configuration:**
- Operating system: [Windows 11](https://www.microsoft.com/en-us/windows/windows-11) or later;
- C++ compiler: [Visual Studio](https://visualstudio.microsoft.com) 2022 or later;
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 12.2 or later;
- [CMake](https://cmake.org/download/) 3.28 or later;
- [OpenCV](https://opencv.org/releases/) 4.9 or later.
- [Git](https://git-scm.com/downloads) 2.43 or later.

## 3 Running the Program

Launch the motion blur filter application PixiCUDA by running the PixiCUDA.exe file, and you will be presented with the
main window of the application.

### 3.1 The main page of the PixiCUDA

<img src="git_images/..." alt="PICTURE_MAIN" width="600"/>

#### 3.1.1 An example of ...

#### 3.1.2 An example of ...

#### 3.1.3 An example of ...

...

## 4 License

The project is licensed under the [CC BY-NC 4.0 License](LICENSE.md).
