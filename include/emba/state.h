 #pragma once

#include <vector>
#include <utility>
#include <ros/time.h>
#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace EMBA {

typedef Eigen::Matrix<double,1,6> RowVector6d; // 1X6
typedef Eigen::Matrix<double,1,2> RowVector2d; // 1X2
typedef Eigen::Matrix<double,1,3> RowVector3d; // 1X3
typedef Eigen::Matrix<double,2,1> ColVector2d; // 2X1
//typedef Eigen::Matrix<double,2,6> Matx26d; // 2X6

/// In our matlab code, we have two data structures: event_map (sensor size) and ev_warp_map (pano map size),
/// where the derivatives and control pose indexes of each event are saved twice,
/// because each event shows up twice in the measurements: once as ev_curr, once as ev_prev.
/// There exists memory wasting.
///
/// Could we discard ev_warp_map, and only use event_map to form the normal equation?
/// In this way, the data association betweem ev_curr and ev_prev is maintain automatically,
/// which are saved consectively at each pixel.
/// Therefore, we do not need to save any data for ev_prev any more.
/// We just need to save more data in event_map:
/// 1. ev_idx; 2. idx_cp (two, but save one); 3. pm; 4. dM_ddrot_cp (1X6); --> define a struct
///
/// When forming the normal equation, we just loop through the event_map,
/// every two consecutive events are paired naturally to be one measurement.
/// Then we can just count the active pixels on the pano map (using a map),
/// and form the normal equation in a cummulative way.
///

// The states of a warped event for OEGM
struct State_OEGM
{
    // The index of this inlier event (set to -1, if it is outlier)
    int inlier_idx; // int (4 bytes): -2,147,483,648 to 2,147,483,647
    // The polarity of this event {neg:0, pos:1}
    int pol;
    // The timestamp of this event
    ros::Time ts; // Not needed for BA
    // The pixel location on the pano map after being warped
    cv::Point2i pm; // uint32_t (4 bytes): 0 to 4,294,967,295
    // Every event involves two consecutive control poses (linear spline): {cp_1st, cp_2nd}
    // Here we just save the index of the first one, then we can get that for the second one by +1
    // The index of the first involved control pose of this event
    uint16_t cp_idx; // uint16_t (2 bytes): 0 to 65,535
    // The derivative of the pixel intensity M(x) with respect to
    // the peturbation on the neighboring control poses {cp_1st, cp_2nd}
    RowVector6d dM_ddrot_cp; // 1X6
};

// The states of a warped event for LEGM
struct State_LEGM
{
    // The index of this event
    int inlier_idx; // int (4 bytes): -2,147,483,648 to 2,147,483,647
    // The polarity of this event {neg:0, pos:1}
    int pol;
    // The timestamp of this event
//    ros::Time ts; // Not needed for BA
    // The pixel location on the pano map after being warped
    cv::Point2d pm;
    // Every event involves two consecutive control poses (linear spline): {cp_1st, cp_2nd}
    // Here we just save the index of the first one, then we can get that for the second one by +1
    // The index of the first involved control pose of this event
    uint16_t cp_idx; // uint16_t: 0 to 65,535
    // The derivative of the pixel location of the warped event with respect to
    // the peturbation on the neiborghing control poses {cp_1st, cp_2nd}
    Eigen::Matrix<double,2,6> dpm_ddrot_cp; // 1X6
    // The map gradient at the pixel location of the warped event
    RowVector2d Gpm;
    // The displacement between the pixel location of this event with the previous event
    ColVector2d dp;
    // The intermediate matrix that is needed by the computation of
    // at the pixel location of the warped event
    RowVector2d temp;
//    // The derivative of the measurement model (pixel intensity change) M(x) with respect to
//    // the peturbation on this pixel (pm) of the gradient map (Gx,Gy)
//    double dM_dGx, dM_dGy;
};

}
