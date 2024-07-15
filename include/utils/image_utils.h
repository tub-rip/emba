#pragma once

#include <opencv2/core/core.hpp>

namespace image_util
{

void minMaxLocRobust(const cv::Mat& image, double& rmin, double& rmax,
                     const double& percentage_pixels_to_discard);

void normalizeRobust(const cv::Mat& src, cv::Mat& dst, const double& percentage_pixels_to_discard);

void saveImgBin(const cv::Mat& img, const std::string& filename);

} // namespace

