#include <iostream>
#include <camera_info_manager/camera_info_manager.h>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

#include "utils/event_pano_warper.h"
#include "utils/pose_manager.h"
#include "utils/rosbag_loading.h"

int main(int argc, char *argv[])
{
    // Dataset Info
    const std::string dataset = "ESIM_simulator";
    const std::string sequence = "synth1";

    // Load groundtruth pano_map from the disk
    const std::string pano_map_path = "/home/shuang/datasets/ESIM_simulator/"
                                      "synth1/reconstructed_image_log_scale.jpg";
    cv::Mat pano_map = cv::imread(pano_map_path, cv::IMREAD_COLOR);
//    cv::imshow("pano_map", pano_map);
//    cv::waitKey(0);

    // Load groundtruth poses from txt files
    utils::PoseManager GT_pose_manager;
    GT_pose_manager.loadPoses(dataset, sequence);

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
//    std::cout << "t_pose_bgn = " << pose_subset.begin()->first.toSec() << std::endl;
//    std::cout << "t_pose_end = " << pose_subset.rbegin()->first.toSec() << std::endl;

    // Initialize the camera trajectory
    LinearTrajectory cam_traj(pose_subset, traj_config);

    /// Warp events on to the map, and visualize them
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
        // Query the pose at the middle point from the camera trajectory
        const::Sophus::SO3d rot = cam_traj.evaluate(t_batch);

        // Loop through all events in this batch
        for (int k = ev_bgn_idx; k < ev_end_idx; k++)
        {
            // Warp events on to the pano map
            cv::Point2i pt_in(events.at(k).x, events.at(k).y);
            cv::Point2d pm = pano_event_warper.warpEventToMap(pt_in, rot);

            // Draw the events on the map
            if (2*events.at(k).polarity-1 > 0)
                // Draw the positive events in red
                cv::circle(pano_map, pm, 1, cv::Scalar(0,0,255), cv::FILLED);
            else
                // Draw the negative events in blue
                cv::circle(pano_map, pm, 1, cv::Scalar(255,0,0), cv::FILLED);
        }
    }

    // Display the pano map with warped events
    cv::imshow("warped_events", pano_map);
    cv::waitKey(0);

    return 0;
}
