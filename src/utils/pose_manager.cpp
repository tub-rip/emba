#include <glog/logging.h>
#include <fstream>
#include "utils/pose_manager.h"

namespace utils {

bool PoseManager::loadPoses(const std::string& filename,
                            const double time_offset)
{
    // Build mapping t <-> T_world_body
    std::ifstream ss;
    ss.open(filename);
    if(!ss.is_open())
    {
        LOG(FATAL) << "Failed to load trajectory file " << filename;
        return false;
    }
    poses_.clear();
    int count = 0;
    std::string str;
    while(std::getline(ss, str))
    {
        std::stringstream ss(str);
        double ts, tx, ty, tz, qx, qy, qz, qw;
        if(ss >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw)
        {
            ts += time_offset; // Apply the time offset
            const Eigen::Quaterniond q(qw, qx, qy, qz);
            const Sophus::SO3d R(q);
            poses_.insert(PoseEntry(ros::Time(ts), R)); // emplace_back() will call the constructor

            count++;
            VLOG(3) << "pose time: " << ros::Time(ts).toSec();
        }
        VLOG(3) << "count poses = " << count;
    }
    ss.close();
    return true;
}

bool PoseManager::loadPoses(const std::string& root_path,
                            const std::string& dataset,
                            const std::string& sequence,
                            const std::string& filename,
                            const double time_offset)
{
    const std::string traj_dir = root_path + "/" + dataset + "/" + sequence + "/traj/";
    // Build mapping t <-> T_world_body
    std::ifstream ss;
    std::string path_traj_file;
    path_traj_file = traj_dir + "interpolation/" + filename + ".txt";

    ss.open(path_traj_file);
    if(!ss.is_open())
    {
        LOG(FATAL) << "Failed to load trajectory file " << path_traj_file;
        return false;
    }
    poses_.clear();
    int count = 0;
    std::string str;
    while(std::getline(ss, str))
    {
        std::stringstream ss(str);
        double ts, tx, ty, tz, qx, qy, qz, qw;
        if(ss >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw)
        {
            ts += time_offset; // Apply the time offset
            const Eigen::Quaterniond q(qw, qx, qy, qz);
            const Sophus::SO3d R(q);
            poses_.insert(PoseEntry(ros::Time(ts), R)); // emplace_back() will call the constructor

            count++;
            VLOG(3) << "pose time: " << ros::Time(ts).toSec();
        }
        VLOG(3) << "count poses = " << count;
    }
    ss.close();
    return true;
}

Sophus::SO3d PoseManager::getPoseAt(const ros::Time& t_query) const
{
    ros::Time t1_, t2_;
    Sophus::SO3d R1_, R2_;

    auto it2 = poses_.upper_bound(t_query);
    if (it2 == poses_.begin())
        return it2->second;
    else if (it2 == poses_.end())
        return poses_.rbegin()->second;
    else
    {
        auto it1 = std::prev(it2);
        t1_ = (it1)->first;
        R1_ = (it1)->second;
        t2_ = (it2)->first;
        R2_ = (it2)->second;

        // Linear interpolation in SO(3)
        auto R_relative = R1_.inverse() * R2_;
        // Interpolation parameter in [0,1]
        auto delta_t = (t_query - t1_).toSec() / (t2_ - t1_).toSec();
        // Linear interpolation, Lie group formulation
        Sophus::SO3d R =  R1_ * Sophus::SO3d::exp(delta_t * R_relative.log());
        return R;
    }
}

PoseMap PoseManager::getPoseSubset(const ros::Time& t1, const ros::Time& t2) const
{
    // Get the head and tail
    auto iter1 = poses_.upper_bound(t1);
    auto iter2 = poses_.lower_bound(t2);

    // Get the pose subset
    PoseMap pose_subset;
    pose_subset.insert(iter1, iter2);
    return pose_subset;
}

}
