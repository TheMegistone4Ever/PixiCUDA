#include <iostream>
#include <opencv2/core/core.hpp>

#include "utils/mae.hpp"
#include "utils/constants.hpp"

long double immae(cv::Mat& i1, cv::Mat& i2)
{
	if (i1.size() != i2.size())
	{
		std::cerr << RED_BOLD << "Images have different sizes..." << RESET_COLOR << std::endl;
		return ERROR_CODE;
	}

	cv::Mat absdiff;
	cv::absdiff(i1, i2, absdiff);
	return static_cast<long double>(cv::sum(absdiff)[0]) / (i1.rows * i1.cols * UCHAR_MAX);
}
