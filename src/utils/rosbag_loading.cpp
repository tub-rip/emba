#include "utils/rosbag_loading.h"

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>
#include <geometry_msgs/PoseStamped.h>
#include <glog/logging.h>

namespace data_loading {

void parse_rosbag(const std::string &rosbag,
                  std::vector<dvs_msgs::Event>& events_,
                  sensor_msgs::CameraInfo& camera_info_msg,
                  const std::string& event_topic,
                  const std::string& camera_info_topic,
                  const double tmin,
                  const double tmax)
{
    std::vector<std::string> topics;
    topics.push_back(event_topic);
    topics.push_back(camera_info_topic);

    events_.clear();

    rosbag::Bag  bag(rosbag, rosbag::bagmode::Read);
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    bool continue_looping_through_bag_ev = true;

    BOOST_FOREACH(rosbag::MessageInstance const m, view)
    {
        if(!continue_looping_through_bag_ev) { break; }

        const std::string& topic_name = m.getTopic();
        VLOG(3) << topic_name;

        // Events
        if (topic_name == topics[0] && continue_looping_through_bag_ev)
        {
            dvs_msgs::EventArray::ConstPtr msg = m.instantiate<dvs_msgs::EventArray>();
            if (msg != NULL)
            {
                if(msg->events.empty()) { continue; }
                for (size_t i = 0; i < msg->events.size(); ++i)
                {
                    const double ev_time_stamp = msg->events[i].ts.toSec();
                    if(ev_time_stamp < tmin) { continue; }
                    if(ev_time_stamp > tmax) { continue_looping_through_bag_ev = false; break; }

                    events_.push_back(msg->events[i]);
                }
            }
        }

        // Camera Info
        if (topic_name == topics[1])
            camera_info_msg = *(m.instantiate<sensor_msgs::CameraInfo>());
    }

    // Sort events by increasing timestamps
    std::sort(events_.begin(), events_.end(),
              [](const dvs_msgs::Event& a, const dvs_msgs::Event& b) -> bool
    {
        return a.ts < b.ts;
    });
}

void parse_rosbag(const std::string &rosbag,
                  std::vector<dvs_msgs::Event>& events_,
                  const std::string& event_topic,
                  const double tmin,
                  const double tmax)
{
    std::vector<std::string> topics;
    topics.push_back(event_topic);

    events_.clear();

    rosbag::Bag  bag(rosbag, rosbag::bagmode::Read);
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    bool continue_looping_through_bag_ev = true;

    BOOST_FOREACH(rosbag::MessageInstance const m, view)
    {
        if(!continue_looping_through_bag_ev) { break; }

        const std::string& topic_name = m.getTopic();
        VLOG(3) << topic_name;

        // Events
        if (topic_name == topics[0] && continue_looping_through_bag_ev)
        {
            dvs_msgs::EventArray::ConstPtr msg = m.instantiate<dvs_msgs::EventArray>();
            if (msg != NULL)
            {
                if(msg->events.empty()) { continue; }
                for (size_t i = 0; i < msg->events.size(); ++i)
                {
                    const double ev_time_stamp = msg->events[i].ts.toSec();
                    if(ev_time_stamp < tmin + 1e-6) { continue; }
                    if(ev_time_stamp > tmax) { continue_looping_through_bag_ev = false; break; }

                    events_.push_back(msg->events[i]);
                }
            }
        }
    }

    // Sort events by increasing timestamps
    std::sort(events_.begin(), events_.end(),
              [](const dvs_msgs::Event& a, const dvs_msgs::Event& b) -> bool
    {
        return a.ts < b.ts;
    });
}

} // namespace data_loading
