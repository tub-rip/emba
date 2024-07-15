#include <iostream>
#include <camera_info_manager/camera_info_manager.h>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

#include "utils/event_pano_warper.h"
#include "utils/pose_manager.h"
#include "utils/rosbag_loading.h"

#include "epba/event_map.h"
#include "utils/so3_funcs.h"
#include <opencv2/core/eigen.hpp>

typedef EPBA::State_OEGM State;
typedef EPBA::EventMap<State> EventMap;

int main(int argc, char *argv[])
{
    // Dataset Info
    const std::string dataset = "ESIM_simulator";
    const std::string sequence = "synth1";

    // Load groundtruth pano_map from the disk
    const std::string pano_map_path = "/home/shuang/datasets/ESIM_simulator/"
                                      "synth1/reconstructed_image_log_scale.jpg";
    cv::Mat pano_map = cv::imread(pano_map_path, cv::IMREAD_COLOR);

    // Load groundtruth poses from txt files
    utils::PoseManager GT_pose_manager;
    GT_pose_manager.loadPoses(dataset, sequence, "cmaxw_traj_interp");

    // Load events from rosbag
    const std::string rosbag_path = "/home/shuang/datasets/"
            + dataset + "/" + sequence + "/events.bag";
    std::cout << "rosbag_path: " << rosbag_path << std::endl;

    const std::string events_topic = "/dvs/events";
    std::vector<dvs_msgs::Event> events;
    const double start_time_ros = 2.0;
    const double stop_time_ros = 2.2;
    data_loading::parse_rosbag(rosbag_path, events, events_topic,
                               start_time_ros, stop_time_ros);
    int num_events = events.size();
    std::cout << "total num events = " << num_events << std::endl;

    // Load camera info
    ros::init(argc, argv, "event_warper_test");
    ros::NodeHandle nh;
    const std::string camera_name = "DVS-synth1";
    std::cout << "camera name: " << camera_name << std::endl;
    camera_info_manager::CameraInfoManager cam_info_manager(nh, camera_name);

    // Initialize event warper
    dvs::EventWarper pano_event_warper;
    int pano_height = pano_map.rows;
    int pano_width = 2 * pano_height;
    pano_event_warper.initialize(cam_info_manager.getCameraInfo(),
                                 pano_width, pano_height);

    // Settings for the camera trajectory
    TrajectorySettings traj_config;
    traj_config.t_beg = ros::Time(2.0);
    traj_config.t_end = ros::Time(2.2);
    traj_config.dt_knots = 0.05;

    // Select the poses within [t_bgn, t_end]
    PoseMap pose_subset = GT_pose_manager.getPoseSubset(traj_config.t_beg, traj_config.t_end);

    // Initialize the camera trajectory
    LinearTrajectory cam_traj(pose_subset, traj_config);

    /// Warp events on to the map, and visualize them
    // Initialize the event map
    EventMap event_map(pano_event_warper.getCameraSize().width,
                       pano_event_warper.getCameraSize().height);

    // Process events in small batch for speed-up
    int ev_batch_size = 100;
    int num_batches = std::ceil(num_events/ev_batch_size);
    // Loop through all the events, and warped them onto the pano map
    for (int i = 0; i < num_batches; i++)
    {
        // Get events (head and tail) in this batch
        int ev_bgn_idx = i * ev_batch_size; // Idx of the first event of this batch
        int ev_end_idx = (i+1) * ev_batch_size; // Idx of the last event of this batch
        // If the index exceeds the total number of events (the last batch)
        if (ev_end_idx > num_events)
        {
            ev_end_idx = num_events;
            ev_batch_size = ev_end_idx - ev_bgn_idx;
        }

        /// Compute the common variables for this event batch
        // Timestamp at the middle point of this batch
        const ros::Time t_batch_bgn = events.at(ev_bgn_idx).ts;
        const ros::Time t_batch_end = events.at(ev_end_idx).ts;
        ros::Duration timespan = t_batch_end - t_batch_bgn;
        ros::Time t_batch = t_batch_bgn + timespan * 0.5;

//        // Query the pose at the middle point from the camera trajectory
//        const::Sophus::SO3d rot = cam_traj.evaluate(t_batch);

        /// TODO: to see if the derivative given by our derivation and that from TUM library
        /// are the same?

        // (1) drot_dcp_perturb (poses w.r.t. perturbations on the control poses) 3X6
        cv::Mat ddrot_ddrot_cp; // drot/ddrot_cp
        int idx_cp_beg;
        const::Sophus::SO3d rot = cam_traj.evaluate(t_batch, &idx_cp_beg, &ddrot_ddrot_cp);

//        std::cout << "idx_cp_beg = " << idx_cp_beg << std::endl;
//        std::cout << "ddrot_ddrot_cp = \n" << ddrot_ddrot_cp << std::endl;

//        const Eigen::Matrix3d Jl = so3_utils::Jl(rot.log());
//        std::cout << "Jl = \n" << Jl << std::endl;
//        Eigen::Matrix3f Jl_copy;
//        Jl_copy << Jl(0,0), Jl(0,1), Jl(0,2),
//                   Jl(1,0), Jl(1,1), Jl(1,2),
//                   Jl(2,0), Jl(2,1), Jl(2,2);
//        std::cout << "Jl_copy = \n" << Jl_copy << std::endl;

//        cv::Matx33f Jl_cv;
//        cv::eigen2cv(Jl_copy, Jl_cv); // J_l = ddrot/drot
//        std::cout << "Jl_cv = \n" << Jl_cv << std::endl;

//        cv::Mat ddrot_ddrot_cp = cv::Mat::zeros(3,6,CV_32FC1);
//        ddrot_ddrot_cp = Jl_cv * drot_ddrot_cp; // ddrot/ddrot_cp = ddrot/drot * drot/ddrot_cp

        std::cout << "(1) TUM library: \n ddrot_ddrot_cp = \n" << ddrot_ddrot_cp << std::endl;

        // (2) our derivation
        // Get the two neighboring control poses
        const Sophus::SO3d ctrl_pose_prev = cam_traj.getControlPose(idx_cp_beg);
        const Sophus::SO3d ctrl_pose_next = cam_traj.getControlPose(idx_cp_beg+1);
        // Compute the rotation increments between them
        const Eigen::Vector3d drotv = (ctrl_pose_next * ctrl_pose_prev.inverse()).log();
        // Get timestamps of these two control poses
        const double t_ctrl_prev = cam_traj.getCtrlPoseTime(idx_cp_beg);
        const double t_ctrl_next = cam_traj.getCtrlPoseTime(idx_cp_beg+1);
//        std::cout << "t_ctrl_prev = " << t_ctrl_prev << std::endl;
//        std::cout << "t_ctrl_next = " << t_ctrl_next << std::endl;
        // Compute the normalized timestamp
        const double u = (t_batch.toSec()-t_ctrl_prev)/cam_traj.getDtCtrlPoses();
//        std::cout << "u = " << u << std::endl;
        // Compute Jl and Jl_u_inv
        Eigen::Matrix3d Jl_inv = so3_utils::Jl_inv(drotv);
        Eigen::Matrix3d Jl_u_inv = so3_utils::Jl(u*drotv);
        Eigen::Matrix3d Au = u * Jl_u_inv * Jl_inv;
        // 3X3 identity matrix
        const Eigen::Matrix3d Id = Eigen::Matrix3d::Identity();
        // compute ddrot_ddrot_cp using chain rule
        Eigen::Matrix3d ddrot_ddrot_cp1 = Id - Au;
        Eigen::Matrix3d ddrot_ddrot_cp2 = Au;
        std::cout << "(2) Our derivation: \n ddrot_ddrot_cp1 = \n " << ddrot_ddrot_cp1
                  << "\n ddrot_ddrot_cp2 = \n" << ddrot_ddrot_cp2 << std::endl;

        std::cout << "********************************************" << std::endl;

        // Loop through all events in this batch
        for (int k = ev_bgn_idx; k < ev_end_idx; k++)
        {
            // Warp events on to the pano map
            cv::Point2i pt_in(events.at(k).x, events.at(k).y);
            cv::Point2d pm = pano_event_warper.warpEventToMap(pt_in, rot, nullptr);

            // Draw the events on the map
            if (2*events.at(k).polarity-1 > 0)
                // Draw the positive events in red
                cv::circle(pano_map, pm, 1, cv::Scalar(0,0,255), cv::FILLED);
            else
                // Draw the negative events in blue
                cv::circle(pano_map, pm, 1, cv::Scalar(255,0,0), cv::FILLED);

            State ev_state;
            // Only update the timestamp in the state (for debugging)
            ev_state.ts = events.at(k).ts;
            // Add this event to the event map
            event_map.addEvent(events.at(k).x, events.at(k).y, ev_state);
        }

        // Show the evolution of the time map
        cv::Mat time_map = event_map.getTimeMap(traj_config.t_beg);
        cv::imshow("time map", time_map);
        // Show the evolution of the event number map
        cv::Mat event_num_map = event_map.getEventNumMap();
        cv::imshow("event number map", event_num_map);
        // Show the evolution of the warped events
        cv::imshow("warped_events", pano_map);
        cv::waitKey(1);
    }

    // Display the pano map with warped events
    cv::imshow("warped_events", pano_map);
    cv::waitKey(0);

    return 0;
}
