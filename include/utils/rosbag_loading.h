#ifndef ROSBAG_LOADING_H
#define ROSBAG_LOADING_H

#include <dvs_msgs/EventArray.h>
#include <camera_info_manager/camera_info_manager.h>

namespace data_loading {

void parse_rosbag(const std::string &rosbag,
                  std::vector<dvs_msgs::Event>& events_,
                  sensor_msgs::CameraInfo& camera_info_msg,
                  const std::string& event_topic,
                  const std::string& camera_info_topic,
                  const double tmin,
                  const double tmax);

void parse_rosbag(const std::string &rosbag,
                  std::vector<dvs_msgs::Event>& events_,
                  const std::string& event_topic,
                  const double tmin,
                  const double tmax);

} // namespace data_loading

#endif // ROSBAG_LOADING_H
