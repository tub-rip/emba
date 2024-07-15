#pragma once

#include <iostream>
#include "utils/image_utils.h"

namespace image_util
{

/**
* \brief Compute robust min and max values (statistics) of an image
* \note Requires sorting
*/
void minMaxLocRobust(const cv::Mat& image, double& rmin, double& rmax,
                     const double& percentage_pixels_to_discard)
{
  cv::Mat image_as_row = image.reshape(0,1);
  cv::Mat image_as_row_sorted;
  cv::sort(image_as_row, image_as_row_sorted, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);
  image_as_row_sorted.convertTo(image_as_row_sorted, CV_64FC1);
  const int single_row_idx_min = (0.5f*percentage_pixels_to_discard/100.f)*image.total();
  const int single_row_idx_max = (1.f - 0.5f*percentage_pixels_to_discard/100.f)*image.total();
  rmin = image_as_row_sorted.at<double>(single_row_idx_min);
  rmax = image_as_row_sorted.at<double>(single_row_idx_max);
}


/**
* \brief Normalize image to the range [0,255] using robust min and max values
*/
void normalizeRobust(const cv::Mat& src, cv::Mat& dst, const double& percentage_pixels_to_discard)
{
  double rmin_val, rmax_val;
  minMaxLocRobust(src, rmin_val, rmax_val, percentage_pixels_to_discard);
  //std::cout << "min_val: " << rmin_val << ", " << "max_val: " << rmax_val << std::endl;
  const double scale = ((rmax_val != rmin_val) ? 255.f / (rmax_val - rmin_val) : 1.f);
  cv::Mat state_image_normalized = scale * (src - rmin_val);
  state_image_normalized.convertTo(dst, CV_8UC1);
}

/**
* \brief Save images as binary files into the disk
*/
void saveImgBin(const cv::Mat& img, const std::string& filename)
{
    FILE* pFile = fopen(filename.c_str(), "wb");
    const size_t total_num_pixels = img.rows * img.cols;
    if (img.type() == CV_64FC1)
    {
        fwrite(img.data, sizeof(double), total_num_pixels, pFile);
    }
    else if (img.type() == CV_32FC1)
    {
        fwrite(img.data, sizeof(float), total_num_pixels, pFile);
    }
    else
    {
        std::cerr << "Image type should be CV_64FC1 or CV_32FC1";
    }
    fclose(pFile);

}

} // namespace
