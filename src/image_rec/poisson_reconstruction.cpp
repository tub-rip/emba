//#include "image_rec/poisson_reconstruction.h"
#include "image_rec/laplace.h"

#include <opencv2/opencv.hpp>
#include <boost/multi_array.hpp>

namespace poisson_reconstruction {

  cv::Mat reconstructFromGradient(const cv::Mat& gradients)
  {
    assert(gradients.type() == CV_64FC2);
    assert(gradients.cols > 0 && gradients.rows > 0);
    const cv::Size img_size = gradients.size();

    // Compute F = dgx/dx + dgy/dy and put it into a boost::multi_array
    const size_t height = img_size.height;
    const size_t width = img_size.width;
    boost::multi_array<double,2> M(boost::extents[height][width]);
    boost::multi_array<double,2> F(boost::extents[height][width]);

    for(size_t i=0; i < height-1; ++i)
    {
      for(size_t j=0; j < width-1; ++j)
      {
        F[i][j] = gradients.at<cv::Vec2d>(i,j+1)[0] - gradients.at<cv::Vec2d>(i,j)[0]
            + gradients.at<cv::Vec2d>(i+1,j)[1] - gradients.at<cv::Vec2d>(i,j)[1];
      }
    }
    F[height-1][width-1] = 0.0;

    // Solve for M_xx + M_yy = F using Poisson solver,
    // with constant intensity boundary conditions
    //const double gradient_on_boundary = 0.0;
    const double intensity_on_boundary = 0.0;
    double trunc;
    trunc = pde::poisolve(M, F, 1.0, 1.0, 1.0, 1.0,
                          intensity_on_boundary,
                          pde::types::boundary::Dirichlet,false);

    cv::Mat reconstructedM(img_size, CV_64F);
    for(size_t i=0; i < height; ++i)
    {
      for(size_t j=0; j < width; ++j)
      {
        reconstructedM.at<double>(i,j) = M[i][j];
      }
    }

    return reconstructedM;
  }

}
