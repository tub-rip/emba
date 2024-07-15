#include "utils/event_pano_warper.h"
#include <opencv2/core/eigen.hpp>
#include <glog/logging.h>

namespace dvs {

void EventWarper::initialize(const sensor_msgs::CameraInfo& camera_info_msg,
                             int pano_width, int pano_height)
{
    // Set camera info
    cam_.fromCameraInfo(camera_info_msg);

    // Set pano map
    pano_width_ = pano_width;
    pano_height_ = pano_height;
    pano_size_ = cv::Size(pano_width_, pano_height_);

    // Equirectangular projection
    pano_cam_ = dvs::EquirectangularCamera(pano_size_, 360.0, 180.0);

    // Pre-compute bearing vectors
    precomputeBearingVectors();

    VLOG(2) << "Event warper initialized";
}

void EventWarper::precomputeBearingVectors()
{
    int sensor_width = cam_.fullResolution().width;
    int sensor_height = cam_.fullResolution().height;

    for(int y=0; y < sensor_height; y++)
    {
        for(int x=0; x < sensor_width; x++)
        {
            cv::Point2d rectified_point = cam_.rectifyPoint(cv::Point2d(x,y));
            cv::Point3d bearing_vec = cam_.projectPixelTo3dRay(rectified_point);
            precomputed_bearing_vectors_.emplace_back(bearing_vec);
        }
    }
}

cv::Point2d EventWarper::warpEventToMap(const cv::Point2i &pt_in,
                                        const Sophus::SO3d& rot,
                                        cv::Matx23d* jacobian)
{
    int sensor_width = cam_.fullResolution().width;
    const int idx = pt_in.y*sensor_width + pt_in.x;

    // Get bearing vector (in look-up-table) corresponding to current event's pixel
    const cv::Point3d bvec = precomputed_bearing_vectors_.at(idx);
    Eigen::Vector3d e_ray_cam(bvec.x, bvec.y, bvec.z);

    // Rotate according to pose(t)
    Eigen::Matrix3d R = rot.matrix();
    Eigen::Vector3d rb = R * e_ray_cam; // rotated bearing vector

    // Project onto the panoramic map
    if (jacobian != nullptr)
    {
        // With computing derivatives
        cv::Matx33d drb_ddrot(0, rb(2), -rb(1), -rb(2), 0, rb(0), rb(1), -rb(0), 0);
        cv::Matx23d dpm_drb;
        Eigen::Vector2d px_mosaic = pano_cam_.projectToImage(rb, &dpm_drb);
        *jacobian = dpm_drb * drb_ddrot;
        return cv::Point2d(px_mosaic[0], px_mosaic[1]);
    }
    else
    {
        // Without computing derivatives
        Eigen::Vector2d px_mosaic = pano_cam_.projectToImage(rb, nullptr);
        return cv::Point2d(px_mosaic[0], px_mosaic[1]);
    }
}

void EventWarper::drawSensorFOV(cv::Mat& canvas,
                                const Sophus::SO3d& R,
                                const cv::MarkerTypes& marker_type,
                                const int marker_size,
                                const cv::Vec3i& color)
{
    int sensor_width = cam_.fullResolution().width;
    int sensor_height = cam_.fullResolution().height;

    // Plot the center of the FOV
    const cv::Point2d warped_center_pt = warpEventToMap(cv::Point2i(sensor_height/2,sensor_width/2), R, nullptr);
    cv::drawMarker(canvas, warped_center_pt, color, marker_type, marker_size, 1);
}

}
