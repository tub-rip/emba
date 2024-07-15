#pragma once

#include "utils/trajectory.h"
#include "utils/equirectangular_camera.h"
#include "emba/event_map.h"
// #include "utils/image_geom_util.h"
// #include "utils/image_utils.h"
// #include "utils/parameters.h"

#include <vector>

#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_geometry/pinhole_camera_model.h>

namespace dvs {

class EventWarper{
public:
    // Constructor
    EventWarper() {}
    ~EventWarper() {}

    // initialize camera and pano info
    void initialize(const sensor_msgs::CameraInfo& camera_info_msg,
                    int pano_width, int pano_height);

    // Precompute bearing vectors to speed up the event warping
    void precomputeBearingVectors();

    // Equirectangular projection
    // Warp an event onto the map, while computing the derivatives
    cv::Point2d warpEventToMap(const cv::Point2i& pt_in, const Sophus::SO3d& rot,
                               cv::Matx23d* dpm_dpw);

    // Draw the FOV of the event camera
    void drawSensorFOV(cv::Mat& canvas,
                       const Sophus::SO3d& rot,
                       const cv::MarkerTypes& marker_type,
                       const int marker_size,
                       const cv::Vec3i& color);

    // Return camera size
    cv::Size getCameraSize() const { return cam_.fullResolution(); }

    // Return pano map size
    int panoMapWidth() const { return pano_width_; }
    int panoMapHeight() const { return pano_height_; }

private:
    // Pinhole camera model
    image_geometry::PinholeCameraModel cam_;

    // Map (panorama) size
    int pano_width_, pano_height_;
    cv::Size pano_size_;

    // Pano camera model for equirectangular projection
    dvs::EquirectangularCamera pano_cam_;

    // Precomputed bearing vector
    std::vector<cv::Point3d> precomputed_bearing_vectors_;
};

}
