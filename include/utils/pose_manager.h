#pragma once

#include <ros/ros.h>
#include <sophus/so3.hpp>

typedef std::pair<ros::Time, Sophus::SO3d> PoseEntry;
typedef std::map<ros::Time, Sophus::SO3d> PoseMap;

namespace utils {

class PoseManager
{
public:
    // Constructor
    PoseManager () {}
    // Deconstructor
    ~PoseManager() {}

    // Load poses from txt files
    bool loadPoses(const std::string& root_path,
                   const std::string& dataset,
                   const std::string& sequence,
                   const std::string& filename,
                   const double time_offset);

    bool loadPoses(const std::string& filename,
                   const double time_offset);

    // Query the (rotational component) pose at some time of interest
    Sophus::SO3d getPoseAt(const ros::Time& t_query) const;

    // Get a subset of poses from a specified time interval [t1,t2]
    PoseMap getPoseSubset(const ros::Time& t1, const ros::Time& t2) const;

private:

    // The sequences of the poses
    PoseMap poses_;
};

}
