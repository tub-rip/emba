#pragma once

#include <map>
#include <array>
#include <assert.h>

#include <opencv2/core.hpp>
#include <ros/ros.h>
#include <Eigen/Core>

#include "basalt/spline/so3_spline.h"

typedef std::pair<ros::Time, Sophus::SO3d> PoseEntry;
typedef std::map<ros::Time, Sophus::SO3d> PoseMap;
typedef std::vector<PoseEntry> PoseArray;

struct TrajectorySettings{
    ros::Time t_beg;
    ros::Time t_end;
    double dt_knots;
};

class Trajectory
{
public:
    // Virtual deconstructor
    virtual ~Trajectory() {}

    // Get timestampes for the start and the end of the trajectory
    ros::Time begTime() const { return ros::Time(t_beg_); }

    // Get the time gap between consecutive control poses
    double getDtCtrlPoses() const { return dt_knots_; }

    virtual size_t size() = 0;
    virtual void resize(size_t n) = 0;

    virtual int NumInvolvedControlPoses() const = 0;

    // Print information of this trajectory
    virtual void print() = 0;

    // Write this trajectory into the disk
    virtual void write(const std::string filename, const double time_offset) = 0;

    // Get or set the control point at given index
    virtual Sophus::SO3d getControlPose(const int idx) = 0;

    // Add multiple new control points at the tail of the trajectory (growing)
    virtual void pushbackCtrlPoses(const std::vector<Sophus::SO3d> &cps) = 0;

    // Evaluate trajectory at some timestamp
    virtual Sophus::SO3d evaluate(const ros::Time& t, int* idx_beg = nullptr,
                                  cv::Mat* jacobian = nullptr) = 0;

    virtual void incrementalUpdate(const std::vector<Eigen::Vector3d>& drotv, const int& idx_beg) = 0;

    // Create a copy of this trajectory and return its pointer
    virtual Trajectory* clone() = 0;

    // Create a copy of a segment of this trajectory and return its pointer
    // The segment is defined by the indexes of control poses [idx_cp_beg, idx_cp_end)
    virtual Trajectory* cloneSegment(const size_t idx_cp_beg, const size_t idx_cp_end) = 0;

    // Copy a segment of traj_src [idx_cp_beg_src, idx_cp_beg_src + num_cp_copy)
    // to this trajectory [idx_cp_beg_dst, idx_cp_beg_dst + num_cp_copy) ==> replace the original control poses
    virtual void replaceWith(Trajectory* traj_src, const size_t num_cp_copy,
                             const size_t idx_cp_beg_src, const size_t idx_cp_beg_dst) = 0;

    virtual Trajectory* CopyAndIncrementalUpdate(const std::vector<Eigen::Vector3d>& drotv,
                                                 const int& idx_traj_beg,
                                                 const int& idx_opt_beg) = 0;

    // Generate control poses with new poses and time interval (mostly used for trajectory extension)
    virtual std::vector<Sophus::SO3d> generateCtrlPoses(PoseMap& poses,
                                                        const ros::Time& t_beg,
                                                        const ros::Time& t_end) = 0;

    // Generate control poses for long trajectories (first split and then merge)
    /// This is needed because the "lift-solve-retract" only works,
    /// when pose increments (perturbations) are small.
    /// In this method, we split the long time interval into small sub-intervals,
    /// and generate control poses for each sub-interval and then merge them
    virtual std::vector<Sophus::SO3d> generateCtrlPosesLong(PoseMap& poses,
                                                            const ros::Time& t_beg,
                                                            const ros::Time& t_end,
                                                            const double sub_interval_length) = 0;

protected:
    // Trajectory configuration
    double t_beg_, dt_knots_;
    int64_t t_beg_ns_, dt_knots_ns_;

    /* Spline initializatoin */
    // Use given poses to compute initial control poses for the trajectory
    virtual void initializeCtrlPoses(PoseMap& poses,
                                     const ros::Time& t_beg,
                                     const ros::Time& t_end) = 0;
    // Fit control poses with the given poses (mostly called by the above functions)
    virtual std::vector<Sophus::SO3d> fitCtrlPoses(PoseMap& poses, const double t_beg, const int num_cps) = 0;

    // Interpolate a pose between two poses at the middle time point
    PoseEntry interpPoseMid(const PoseEntry& p1, const PoseEntry& p2);
};

class LinearTrajectory: public Trajectory
{
public:
    typedef typename basalt::So3Spline<2, double> SO3_Spline_Traj;
    typedef typename basalt::So3Spline<2, double>::JacobianStruct Jacobian_wrt_Control_Points;

    // Constructor
    LinearTrajectory(const TrajectorySettings& traj_config); // Create an empty trajectory (with no control poses)
    LinearTrajectory(PoseMap &poses, const TrajectorySettings& traj_config);
    LinearTrajectory(const double t_beg, const double dt_knots,
                     const std::vector<Sophus::SO3d>& ctrl_poses);

    // Deconstructor
    ~LinearTrajectory() { }

    // Get of change the number of control points (for re-initialization)
    size_t size() override { return spline_.size(); }
    void resize(size_t n) override { spline_.resize(n); }

    // The number of control poses that are required to interpolate a pose
    int NumInvolvedControlPoses() const override { return 2; }

    // Print information of this trajectory
    void print() override;

    // Write this trajectory into the disk
    void write(const std::string filename, const double time_offset) override;

    // Get or set the control point at given index
    Sophus::SO3d getControlPose(const int idx) override;

    // Get timestamp of the control point
    double getCtrlPoseTime(const int idx) const { return t_beg_ + idx*dt_knots_; }

    // Add new control points at the tail of the trajectory (growing)
    void pushbackCtrlPoses(const std::vector<Sophus::SO3d> &cps) override;

    // Query pose at given timestamp
    Sophus::SO3d evaluate(const ros::Time &t, int* idx_beg = nullptr,
                          cv::Mat *jacobian = nullptr) override; // jaocbian: 3X6

    // Update the trajectory by incremental rotation (delta R)
    void incrementalUpdate(const std::vector<Eigen::Vector3d>& drotv, const int& idx_beg) override;

    // Create a copy of this trajectory and return its pointer
    LinearTrajectory* clone() override;

    // Create a copy of a segment of this trajectory and return its pointer
    // The segment is defined by the indexes of control poses [idx_cp_beg, idx_cp_end)
    LinearTrajectory* cloneSegment(const size_t idx_cp_beg, const size_t idx_cp_end) override;

    // Copy a segment of traj_src [idx_cp_beg_src, idx_cp_beg_src + num_cp_copy)
    // to this trajectory [idx_cp_beg_dst, idx_cp_beg_dst + num_cp_copy) ==> replace the original control poses
    void replaceWith(Trajectory* traj_src, const size_t num_cp_copy,
                     const size_t idx_cp_beg_src, const size_t idx_cp_beg_dst) override;

    // Return a new trajectory that is updated from (a part of) the current trajectory using incremental rot_vec
    LinearTrajectory* CopyAndIncrementalUpdate(const std::vector<Eigen::Vector3d>& drotv,
                                               const int& idx_traj_beg,
                                               const int& idx_opt_beg) override;

    // Generate control poses with new poses and time interval (mostly used for trajectory extension)
    std::vector<Sophus::SO3d> generateCtrlPoses(PoseMap& poses,
                                                        const ros::Time& t_beg,
                                                        const ros::Time& t_end) override;

    // Generate control poses for long trajectories (first split and then merge)
    std::vector<Sophus::SO3d> generateCtrlPosesLong(PoseMap& poses,
                                                    const ros::Time& t_beg,
                                                    const ros::Time& t_end,
                                                    const double sub_interval_length) override;

protected:
    // Initialize control poses for this trajectory, with the poses on the trajectory
    void initializeCtrlPoses(PoseMap& poses,
                             const ros::Time& t_beg,
                             const ros::Time& t_end) override;

    // Fit control poses with the given poses (mostly called by the above functions)
    virtual std::vector<Sophus::SO3d> fitCtrlPoses(PoseMap& poses, const double t_beg, const int num_cps) override;

private:
    // Trajectory (linear spline)
    SO3_Spline_Traj spline_;
};

class CubicTrajectory: public Trajectory
{
public:
    typedef typename basalt::So3Spline<4, double> SO3_Spline_Traj;
    typedef typename basalt::So3Spline<4, double>::JacobianStruct Jacobian_wrt_Control_Points;

    // Constructor
    CubicTrajectory(const TrajectorySettings& traj_config); // Create an empty trajectory (with no control poses)
    CubicTrajectory(PoseMap &poses, const TrajectorySettings& traj_config);
    CubicTrajectory(const double t_beg, const double dt_knots,
                    const std::vector<Sophus::SO3d>& ctrl_poses);

    //    CubicTrajectory(int64_t dt_knots_ns, int64_t t_beg_ns, std::vector<Sophus::SO3d>& control_poses);

    ~CubicTrajectory() { }

    // Get of change the number of control points (for re-initialization)
    size_t size() override { return spline_.size(); }
    void resize(size_t n) override { spline_.resize(n); }

    int NumInvolvedControlPoses() const override { return 4; }

    // Print information of this trajectory
    void print() override;

    // Write this trajectory into the disk
    void write(const std::string filename, const double time_offset) override;

    // Get or set the control point at given index
    Sophus::SO3d getControlPose(const int idx) override;

    // Add new control points at the tail of the trajectory (growing)
    // Do not change t_end !!!
    void pushbackCtrlPoses(const std::vector<Sophus::SO3d> &cps) override;

    // Query pose at given timestamp
    Sophus::SO3d evaluate(const ros::Time &t, int* idx_beg = nullptr,
                          cv::Mat *jacobian = nullptr) override; // jacobian: 3X12

    // Update the trajectory by incremental rotation (delta R)
    void incrementalUpdate(const std::vector<Eigen::Vector3d>& drotv, const int& idx_beg) override;

    // Create a copy of this trajectory and return its pointer
    CubicTrajectory* clone() override;

    // Create a copy of a segment of this trajectory and return its pointer
    // The segment is defined by the indexes of control poses [idx_cp_beg, idx_cp_end)
    CubicTrajectory* cloneSegment(const size_t idx_cp_beg, const size_t idx_cp_end) override;

    // Copy a segment of traj_src [idx_cp_beg_src, idx_cp_beg_src + num_cp_copy)
    // to this trajectory [idx_cp_beg_dst, idx_cp_beg_dst + num_cp_copy) ==> replace the original control poses
    void replaceWith(Trajectory* traj_src, const size_t num_cp_copy,
                     const size_t idx_cp_beg_src, const size_t idx_cp_beg_dst) override;

    // Return a new trajectory that is updated from the current trajectory using incremental rot_vec
    CubicTrajectory* CopyAndIncrementalUpdate(const std::vector<Eigen::Vector3d>& drotv,
                                              const int& idx_traj_beg,
                                              const int& idx_opt_beg) override;

    // Generate control poses with new poses and time interval (mostly used for trajectory extension)
    std::vector<Sophus::SO3d> generateCtrlPoses(PoseMap& poses,
                                                const ros::Time& t_beg,
                                                const ros::Time& t_end) override;

    // Generate control poses for long trajectories (first split and then merge)
    std::vector<Sophus::SO3d> generateCtrlPosesLong(PoseMap& poses,
                                                    const ros::Time& t_beg,
                                                    const ros::Time& t_end,
                                                    const double sub_interval_length) override;

protected:
    // Initialize control poses for this trajectory, with the poses on the trajectory
    void initializeCtrlPoses(PoseMap& poses,
                             const ros::Time& t_beg,
                             const ros::Time& t_end) override;
    // Fit control poses with the given poses (mostly called by the above functions)
    virtual std::vector<Sophus::SO3d> fitCtrlPoses(PoseMap& poses, const double t_beg, const int num_cps) override;

private:
    // Trajectory (cubic spline)
    SO3_Spline_Traj spline_;
};
