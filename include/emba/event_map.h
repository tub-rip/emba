#pragma once

#include <vector>
#include <utility>

#include <ros/time.h>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

#include <opencv2/imgproc.hpp>

#include "emba/state.h"

namespace EMBA {

/// Keeps track of the events occuring at each pixel
/// and the state of the system when the events were triggered.

/// Each pixel has a vector to store the states of all the events warped here,
/// which would be used to form the normal equation.

template<typename State> class EventMap
{
public:
    // Constructor
    EventMap() {}
    EventMap(int width, int height)
        : width_(width)
        , height_(height)
        , map_(std::vector<std::vector<State>>(width * height))
    {}

    // Add an event (and its state) into the event map
    void addEvent(int x, int y, const State& state)
    {
        map_[pxToIdx(x,y)].emplace_back(state);
    }

    // Get size of the event map
    int width() const { return width_; }
    int height() const { return height_; }

    // Return (reference to, avoid copying) all the states saved in this pixel.
    std::vector<State>& at(int x, int y)
    {
        return map_[pxToIdx(x, y)];
    }

    // Clear all the events
    void clear()
    {
        for (int y = 0; y < height_; y++)
        {
            for (int x = 0; x < width_; x++)
            {
                this->at(x,y).clear();
            }
        }
    }

    // Return the corresponding time map (for debugging)
    cv::Mat getTimeMap(const ros::Time& t0) const
    {
        cv::Mat time_map = cv::Mat::zeros(height_, width_, CV_64FC1);
        // Loop through all the pixels,
        // and write the timestamp of the latest event into the time map
        for (int x = 0; x < width_; x++)
        {
            for (int y = 0; y < height_; y++)
            {
                double ts_latest;
                if (map_[pxToIdx(x, y)].empty())
                    ts_latest = t0.toSec();
                else
                    ts_latest = map_[pxToIdx(x, y)].rbegin()->ts.toSec();

                time_map.at<double>(cv::Point2i(x,y)) = ts_latest;
            }
        }
        // Normalize to [0,255] for display
        cv::normalize(time_map, time_map, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        return time_map;
    }

    // Return the map of event numbers (for debugging)
    cv::Mat getEventNumMap() const
    {
        // Loop through all the pixels,
        // write the number of events at each pixel into the time map
        cv::Mat event_num_map = cv::Mat::zeros(height_, width_, CV_32FC1);
        for (int x = 0; x < width_; x++)
        {
            for (int y = 0; y < height_; y++)
            {
                event_num_map.at<float>(cv::Point2i(x,y)) = map_[pxToIdx(x, y)].size();
            }
        }
        // Normalize to [0,255] for display
        cv::normalize(event_num_map, event_num_map, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::applyColorMap(event_num_map, event_num_map, cv::COLORMAP_JET);
        return event_num_map;
    }

private:
    // Convert 2D pixel locations to 1D index
    inline size_t pxToIdx(int x, int y) const { return x + y * width_; }

    // Sensor size
    int width_, height_;

    // 1D row-major data storage
    std::vector<std::vector<State>> map_;
};

}
