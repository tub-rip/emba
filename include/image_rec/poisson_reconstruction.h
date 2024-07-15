#pragma once

#include "opencv2/opencv.hpp"

namespace poisson_reconstruction {

  // boundary condition: zeros on the borders
  cv::Mat reconstructFromGradient(const cv::Mat& gradients);

}
