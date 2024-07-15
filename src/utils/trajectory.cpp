#include "utils/trajectory.h"
#include <math.h>
#include <glog/logging.h>
#include <fstream>

/************************ Spline Base *********************************/
PoseEntry Trajectory::interpPoseMid(const PoseEntry& p1, const PoseEntry& p2)
{
    // Load two poses
    const ros::Duration dt = p2.first - p1.first;
    ros::Time t_mid = p1.first + dt * 0.5;
    const Sophus::SO3d R1 = p1.second;
    const Sophus::SO3d R2 = p2.second;

    // Linear interpolation in SO(3)
    const Sophus::SO3d dR = R1.inverse() * R2;
    // Linear interpolation at the middle time point, Lie group formulation
    const Sophus::SO3d R_mid = R1 * Sophus::SO3d::exp(0.5 * dR.log());
    return PoseEntry(t_mid, R_mid);
}


/*********************** Linear Spline ********************************/
LinearTrajectory::LinearTrajectory(const TrajectorySettings& config)
    :spline_(SO3_Spline_Traj(int64_t(1e9 * config.dt_knots),
                             config.t_beg.toNSec()))
{
    // Get trajectory configuration
    t_beg_ = config.t_beg.toSec();
    //    t_end_ = config.t_end.toSec();

    dt_knots_ = config.dt_knots;
    t_beg_ns_ = config.t_beg.toNSec();
    dt_knots_ns_ = 1e9 * dt_knots_;

    VLOG(4) << "New linear trajectory generated: "
            << "t_traj_beg = " << t_beg_
            << ", dt_knots = " << dt_knots_;
}

LinearTrajectory::LinearTrajectory(PoseMap &poses, const TrajectorySettings& config)
    :spline_(SO3_Spline_Traj(int64_t(1e9 * config.dt_knots),
                             config.t_beg.toNSec()))
{
    // Get trajectory configuration
    t_beg_ = config.t_beg.toSec();
    //    t_end_ = config.t_end.toSec();

    dt_knots_ = config.dt_knots;
    t_beg_ns_ = config.t_beg.toNSec();
    dt_knots_ns_ = 1e9 * dt_knots_;

    VLOG(4) << "New linear trajectory generated: "
            << "t_traj_beg = " << t_beg_
            << ", dt_knots = " << dt_knots_;

    // Initialize control poses with sampled poses on the spline
    initializeCtrlPoses(poses, config.t_beg, config.t_end);
}

LinearTrajectory::LinearTrajectory(const double t_beg, const double dt_knots,
                                   const std::vector<Sophus::SO3d>& ctrl_poses)
    :spline_(SO3_Spline_Traj(int64_t(1e9 * dt_knots),
                             int64_t(1e9 * t_beg)))
{
    // Set trajectory
    t_beg_ = t_beg;
    t_beg_ns_ = int64_t(1e9*t_beg);
    dt_knots_ = dt_knots;
    dt_knots_ns_ = int64_t(1e9*dt_knots);

    // Add control poses
    this->pushbackCtrlPoses(ctrl_poses);
}

Sophus::SO3d LinearTrajectory::getControlPose(const int idx)
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, this->size());
    return spline_.getKnot(idx);
}

void LinearTrajectory::print()
{
    VLOG(0) << "----------------------------------------";
    VLOG(0) << "Linear trajectory";
    VLOG(0) << "t_beg = " << std::setprecision(16) << t_beg_
            << ", dt_knots = " << dt_knots_;
    VLOG(0) << "Control poses:";
    for (size_t i = 0; i < this->size(); ++i)
    {
        VLOG(0) << "#" << i << "= "
                << this->getControlPose(i).log().transpose() << std::endl;
    }
    VLOG(0) << "----------------------------------------";
}

void LinearTrajectory::write(const std::string filename, const double time_offset)
{
    // Write the refined control poses into the txt file (in the format of quaternion)
    std::ofstream writer;
    writer.open(filename);
    for (size_t i = 0; i < this->size(); i++)
    {
        const double t_cp = this->t_beg_ + i * this->dt_knots_;
        const Sophus::SO3d cp = this->getControlPose(i);
        const Eigen::Quaterniond quat = cp.unit_quaternion();
        writer << t_cp - time_offset << " "
               << 0.0 << " " << 0.0 << " " << 0.0 << " "
               << quat.x() << " " << quat.y() << " "
               << quat.z() << " " << quat.w() << std::endl;
    }
    writer.close();
}

void LinearTrajectory::pushbackCtrlPoses(const std::vector<Sophus::SO3d> &cps)
{
    for (auto const& p : cps) { spline_.knotsPushBack(p); }
}


Sophus::SO3d LinearTrajectory::evaluate(const ros::Time &t, int* start_idx,
                                        cv::Mat *jacobian)
{
    int64_t nsec = t.toNSec();
    Sophus::SO3d R;
    if (jacobian != nullptr && start_idx != nullptr)
    {
        *jacobian = cv::Mat::zeros(3, 6, CV_64FC1);
        Jacobian_wrt_Control_Points J;
        R = spline_.evaluate(nsec, &J);
        *start_idx = J.start_idx;

        // For linear spline, jaocbian is 3X6
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                jacobian->at<double>(j, i)   = J.d_val_d_knot[0](j, i);
                jacobian->at<double>(j, i+3) = J.d_val_d_knot[1](j, i);
            }
        }
    }
    else { R = spline_.evaluate(nsec, nullptr); }

    return R;
}

std::vector<Sophus::SO3d> LinearTrajectory::fitCtrlPoses(PoseMap& poses,
                                                         const double t_beg,
                                                         const int num_cps)
{
    CHECK_GE(poses.size(), num_cps); // ensure the equation to be solvable

    /// 1. Lift: go to the tangent space (rotation increment/perturbation)
    // Compute the rotation increment w.r.t. the first pose (offset)
    Sophus::SO3d offset = poses.begin()->second;
    Sophus::SO3d offset_inv = offset.inverse();
    PoseArray poses_incre;
    poses_incre.clear();
    for (auto const& p : poses)
    {
        Sophus::SO3d drot = offset_inv * p.second;
        poses_incre.emplace_back(p.first, drot);
    }
    CHECK_EQ(poses_incre.size(),poses.size());

    /// 2. Solve: solve for the control poses through a linear system in tangent space
    /// https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/INT-APP/CURVE-INT-global.html
    // Pre-allocate memory for matrices
    // Parameter Matrix N
    Eigen::MatrixXd N = Eigen::MatrixXd::Zero(poses_incre.size(), num_cps);

    // Data Matrix D (num_poses X 3)
    Eigen::VectorXd Dx(poses_incre.size());
    Eigen::VectorXd Dy(poses_incre.size());
    Eigen::VectorXd Dz(poses_incre.size());

    // Variable Matrix P (num_control_points X 3)
    Eigen::VectorXd Px(num_cps);
    Eigen::VectorXd Py(num_cps);
    Eigen::VectorXd Pz(num_cps);

    // Basis matrix for uniform linear spline
    Eigen::Matrix2d M2 = (Eigen::MatrixXd(2, 2) << 1.0, 0.0,
                          -1.0, 1.0).finished();

    // Build the linear system (compute the matrices)
    int pose_idx = 0;
    for (auto const& dp: poses_incre)
    {
        // Compute matrix N
        double t = dp.first.toSec();
        // Index of the first control pose that affects p(t)
        int t_i = std::floor((t - t_beg)/dt_knots_);
        // u = (t - t_i) / (t_{i+1} - t_i)s
        double u = (t - (t_i * dt_knots_ + t_beg))/dt_knots_;

        Eigen::Matrix<double, 1, 2> U;
        for (int i = 0; i < 2; i++) { U(i) = std::pow(u, i); }

        Eigen::Matrix<double, 1, 2> N_idx = U * M2;
        for (int j = 0; j < 2; j++) { N(pose_idx, t_i+j) = N_idx(j); }

        // Set Matrix D
        Eigen::Vector3d rot_vec_idx = dp.second.log();
        Dx(pose_idx) = rot_vec_idx(0);
        Dy(pose_idx) = rot_vec_idx(1);
        Dz(pose_idx) = rot_vec_idx(2);

        pose_idx++;
    }

    // Solve NP = D for P (ordered control points)
    Px = N.fullPivHouseholderQr().solve(Dx);
    Py = N.fullPivHouseholderQr().solve(Dy);
    Pz = N.fullPivHouseholderQr().solve(Dz);

    /// 3. Retract: go back to the Lie group by multiplying the offset
    std::vector<Sophus::SO3d> ctrl_poses;
    for (int i = 0; i < num_cps; ++i)
    {
        Eigen::Vector3d drotv(Px(i), Py(i), Pz(i));
        Sophus::SO3d cp = offset * Sophus::SO3d::exp(drotv); // Left or Right? Need to be tested
        ctrl_poses.emplace_back(cp);
    }

    return ctrl_poses;
}

void LinearTrajectory::initializeCtrlPoses(PoseMap& poses,
                                           const ros::Time& t_beg,
                                           const ros::Time& t_end)
{
    // Compute the number of newly generated control poses
    const double time_interval = (t_end - t_beg).toSec();
    const int num_cps = std::round(time_interval/dt_knots_) + 1;

    // Fit control poses with the given poses
    std::vector<Sophus::SO3d> ctrl_poses = fitCtrlPoses(poses, t_beg_, num_cps);
    // Push back to this trajectory
    this->pushbackCtrlPoses(ctrl_poses);

    VLOG(4) << "New linear trajectory initialized";
}

std::vector<Sophus::SO3d> LinearTrajectory::generateCtrlPoses(PoseMap& poses,
                                                              const ros::Time& t_beg,
                                                              const ros::Time& t_end)
{
    // Compute the number of newly generated control poses
    const int num_cps = std::round((t_end - t_beg).toSec()/dt_knots_) + 1;
    // Fit control poses with the given poses
    std::vector<Sophus::SO3d> ctrl_poses = fitCtrlPoses(poses, t_beg.toSec(), num_cps);
    return ctrl_poses;
}

std::vector<Sophus::SO3d> LinearTrajectory::generateCtrlPosesLong(PoseMap &poses,
                                                                  const ros::Time &t_beg,
                                                                  const ros::Time &t_end,
                                                                  const double sub_interval_length)
{
    const double timespan_poses = (t_end - t_beg).toSec();
    const size_t num_unit_invertal = std::floor(timespan_poses/sub_interval_length + 1e-6);

    // Generate control poses with front-end poses
    std::vector<Sophus::SO3d> ctrl_poses;
    for (size_t i = 0; i < num_unit_invertal; i++)
    {
        const ros::Time t_subset_beg = t_beg + ros::Duration(sub_interval_length) * i;
        const ros::Time t_subset_end = t_subset_beg + ros::Duration(sub_interval_length);

        // Get the pose subset for this sub-interval
        auto iter1 = poses.upper_bound(t_subset_beg);
        auto iter2 = poses.lower_bound(t_subset_end);
        PoseMap pose_subset;
        pose_subset.insert(iter1, iter2);
        std::vector<Sophus::SO3d> cp_subset = this->generateCtrlPoses(
                    pose_subset, t_subset_beg, t_subset_end);

        // If not the first sub-window, remove the first control pose
        if (i != 0)
        {
            cp_subset.erase(cp_subset.begin(), cp_subset.begin()+1);
        }

        // Add control poses into the whole control pose set
        for (auto cp : cp_subset)
        {
            ctrl_poses.push_back(cp);
        }
    }
    return ctrl_poses;
}

void LinearTrajectory::incrementalUpdate(const std::vector<Eigen::Vector3d>& drotv, const int& idx_beg)
{
    CHECK_EQ(idx_beg + drotv.size(), this->size());
    for (size_t i = idx_beg; i < this->size(); ++i)
    {
        // Use left perturbation to update control poses
        spline_.getKnot(i) = Sophus::SO3d::exp(drotv[i-idx_beg]) * spline_.getKnot(i);
    }
}

LinearTrajectory* LinearTrajectory::clone()
{
    std::vector<Sophus::SO3d> ctrl_poses;
    for (size_t i = 0; i < this->size(); i++)
    {
        ctrl_poses.push_back(spline_.getKnot(i));
    }
    LinearTrajectory* traj_clone = new LinearTrajectory(this->t_beg_, this->dt_knots_, ctrl_poses);
    return traj_clone;
}

LinearTrajectory* LinearTrajectory::cloneSegment(const size_t idx_cp_beg, const size_t idx_cp_end)
{
    CHECK_GT(idx_cp_end, idx_cp_beg);
    CHECK_GE(this->size(), idx_cp_end);

    std::vector<Sophus::SO3d> ctrl_poses;
    for (size_t i = idx_cp_beg; i < idx_cp_end; i++)
    {
        ctrl_poses.push_back(spline_.getKnot(i));
    }
    const double t_traj_seg_beg = this->t_beg_ + idx_cp_beg * this->dt_knots_;
    LinearTrajectory* traj_seg = new LinearTrajectory(t_traj_seg_beg, this->dt_knots_, ctrl_poses);
    return traj_seg;
}

void LinearTrajectory::replaceWith(Trajectory* traj_src, const size_t num_cp_copy,
                                   const size_t idx_cp_beg_src, const size_t idx_cp_beg_dst)
{
    CHECK_GE(traj_src->size(), idx_cp_beg_src + num_cp_copy);
    CHECK_GE(this->size(), idx_cp_beg_dst + num_cp_copy);

    // Replace the control poses of this trajectory with the ones from traj_src
    for (size_t i = 0; i < num_cp_copy; i++)
    {
        spline_.getKnot(idx_cp_beg_dst+i) = traj_src->getControlPose(idx_cp_beg_src+i);
    }
}

LinearTrajectory* LinearTrajectory::CopyAndIncrementalUpdate(const std::vector<Eigen::Vector3d>& drotv,
                                                             const int& idx_traj_beg,
                                                             const int& idx_opt_beg)
{
    CHECK_EQ(idx_opt_beg + drotv.size(), this->size());
    CHECK_GE(idx_opt_beg, idx_traj_beg);

    // Copy current trajectory to the temporary trajectory (only the part after idx_beg)
    std::vector<Sophus::SO3d> ctrl_poses_traj_temp;
    for (size_t i = idx_traj_beg; i < this->size(); i++)
    {
        ctrl_poses_traj_temp.push_back(spline_.getKnot(i));
    }
    const double t_traj_temp_beg = this->t_beg_ + idx_traj_beg * this->dt_knots_;
    LinearTrajectory* traj_temp = new LinearTrajectory(t_traj_temp_beg, this->dt_knots_, ctrl_poses_traj_temp);

    CHECK_EQ(drotv.size()+idx_opt_beg-idx_traj_beg, traj_temp->size());

    // Incrementally update it
    traj_temp->incrementalUpdate(drotv, idx_opt_beg-idx_traj_beg);
    return traj_temp;
}

/*********************** Cubic Spline ********************************/

CubicTrajectory::CubicTrajectory(const TrajectorySettings& config)
    :spline_(SO3_Spline_Traj(int64_t(1e9 * config.dt_knots),
                             config.t_beg.toNSec()))
{
    // Get trajectory configuration
    t_beg_ = config.t_beg.toSec();
    //    t_end_ = config.t_end.toSec();
    dt_knots_ = config.dt_knots;

    t_beg_ns_ = config.t_beg.toNSec();
    dt_knots_ns_ = 1e9 * dt_knots_;

    VLOG(4) << "New Cubic Trajectory generated: "
            << "t_traj_beg = " << t_beg_
            << ", dt_knots = " << dt_knots_;
}

CubicTrajectory::CubicTrajectory(PoseMap &poses, const TrajectorySettings& config)
    :spline_(SO3_Spline_Traj(int64_t(1e9 * config.dt_knots),
                             config.t_beg.toNSec()))
{
    // Get trajectory configuration
    t_beg_ = config.t_beg.toSec();
    //    t_end_ = config.t_end.toSec();
    dt_knots_ = config.dt_knots;

    t_beg_ns_ = config.t_beg.toNSec();
    dt_knots_ns_ = 1e9 * dt_knots_;

    VLOG(4) << "New Cubic Trajectory generated: "
            << "t_traj_beg = " << t_beg_
            << ", dt_knots = " << dt_knots_;

    // Initialize new control poses by solving a linear system
    initializeCtrlPoses(poses, config.t_beg, config.t_end);
}

CubicTrajectory::CubicTrajectory(const double t_beg, const double dt_knots,
                                 const std::vector<Sophus::SO3d>& ctrl_poses)
    :spline_(SO3_Spline_Traj(int64_t(1e9 * dt_knots),
                             int64_t(1e9 * t_beg)))
{
    // Set trajectory
    t_beg_ = t_beg;
    t_beg_ns_ = int64_t(1e9*t_beg);
    dt_knots_ = dt_knots;
    dt_knots_ns_ = int64_t(1e9*dt_knots);

    // Add control poses
    this->pushbackCtrlPoses(ctrl_poses);
}

Sophus::SO3d CubicTrajectory::getControlPose(const int idx)
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, this->size());
    return spline_.getKnot(idx);
}

void CubicTrajectory::print()
{
    VLOG(0) << "--------------------------------------";
    VLOG(0) << "Cubic trajectory";
    VLOG(0) << "t_beg = " << t_beg_ << ", dt_knots = " << dt_knots_;
    VLOG(0) << "Control poses:";
    for (size_t i = 0; i < this->size(); ++i)
    {
        VLOG(0) << "#" << i << "= "
                << this->getControlPose(i).log().transpose() << std::endl;
    }
    VLOG(0) << "--------------------------------------";
}

void CubicTrajectory::write(const std::string filename, const double time_offset)
{
    // TODO
}

void CubicTrajectory::pushbackCtrlPoses(const std::vector<Sophus::SO3d> &cps)
{
    for (auto const& p : cps) { spline_.knotsPushBack(p); }
}

Sophus::SO3d CubicTrajectory::evaluate(const ros::Time &t, int *start_idx,
                                       cv::Mat *jacobian)
{
    int64_t nsec = t.toNSec();
    Sophus::SO3d R;
    if (jacobian != nullptr && start_idx != nullptr)
    {
        *jacobian = cv::Mat::zeros(3, 12, CV_64FC1);
        Jacobian_wrt_Control_Points J;
        R = spline_.evaluate(nsec, &J);
        *start_idx = J.start_idx;

        // For cubic spline, jaocbian is 3X12
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                jacobian->at<double>(j, i)   = J.d_val_d_knot[0](j, i);
                jacobian->at<double>(j, i+3) = J.d_val_d_knot[1](j, i);
                jacobian->at<double>(j, i+6) = J.d_val_d_knot[2](j, i);
                jacobian->at<double>(j, i+9) = J.d_val_d_knot[3](j, i);
            }
        }
    }
    else { R = spline_.evaluate(nsec, nullptr); }
    return R;
}

std::vector<Sophus::SO3d> CubicTrajectory::fitCtrlPoses(PoseMap& poses,
                                                        const double t_beg,
                                                        const int num_cps)
{
    CHECK_GE(poses.size(), num_cps); // ensure the equation to be solvable

    /// 1. Lift: go to the tangent space (rotation increment/perturbation)
    // Compute the rotation increment w.r.t. the first pose (offset)
    // Compute the rotation increment w.r.t. the first pose (offset)
    Sophus::SO3d offset = poses.begin()->second;
    Sophus::SO3d offset_inv = offset.inverse();
    PoseArray poses_incre;
    poses_incre.clear();
    for (auto const& p : poses)
    {
        Sophus::SO3d drot = offset_inv * p.second;
        poses_incre.emplace_back(p.first, drot);
    }
    CHECK_EQ(poses_incre.size(),poses.size());

    /// 2. Solve: solve for the control poses through a linear system in tangent space
    /// https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/INT-APP/CURVE-INT-global.html
    // Pre-allocate memory for matrices
    // Parameter Matrix N
    Eigen::MatrixXd N = Eigen::MatrixXd::Zero(poses_incre.size(), num_cps);

    // Data Matrix D (num_poses X 3)
    Eigen::VectorXd Dx(poses_incre.size());
    Eigen::VectorXd Dy(poses_incre.size());
    Eigen::VectorXd Dz(poses_incre.size());

    // Variable Matrix P (num_control_points X 3)
    Eigen::VectorXd Px(num_cps);
    Eigen::VectorXd Py(num_cps);
    Eigen::VectorXd Pz(num_cps);

    // Basis matrix for uniform cubic spline, used for cubic spline initialization
    Eigen::Matrix4d M4 = (Eigen::MatrixXd(4, 4) << 1./6, 2./3, 1./6,  0.0,
                          -0.5,  0.0,  0.5,  0.0,
                          0.5, -1.0,  0.5,  0.0,
                          -1./6,  0.5, -0.5, 1./6).finished();

    // Build the linear system (compute the matrices)
    int pose_idx = 0;
    for (auto const& dp: poses_incre)
    {
        /* Compute matrix N */
        double t = dp.first.toSec();
        // Index of the first control pose that influence p(t)
        int t_i = std::floor((t - t_beg)/dt_knots_);
        // u = (t - t_i) / (t_{i+1} - t_i)s
        double u = (t - (t_i * dt_knots_ + t_beg))/dt_knots_;

        Eigen::Matrix<double, 1, 4> U;
        for (int i = 0; i < 4; i++) { U(i) = std::pow(u, i); }

        Eigen::Matrix<double, 1, 4> N_idx = U * M4;
        for (int j = 0; j < 4; j++) { N(pose_idx, t_i+j) = N_idx(j); }

        /* Set Matrix D */
        Eigen::Vector3d rot_vec_idx = dp.second.log();
        Dx(pose_idx) = rot_vec_idx(0);
        Dy(pose_idx) = rot_vec_idx(1);
        Dz(pose_idx) = rot_vec_idx(2);

        pose_idx++;
    }

    // Solve NP = D for P (ordered control points)
    Px = N.fullPivHouseholderQr().solve(Dx);
    Py = N.fullPivHouseholderQr().solve(Dy);
    Pz = N.fullPivHouseholderQr().solve(Dz);

    /// 3. Retract: go back to the Lie group by multiplying the offset
    std::vector<Sophus::SO3d> ctrl_poses;
    for (int i = 0; i < num_cps; ++i)
    {
        Eigen::Vector3d drotv(Px(i), Py(i), Pz(i));
        Sophus::SO3d cp = offset * Sophus::SO3d::exp(drotv); // Left or Right? Need to be tested
        ctrl_poses.emplace_back(cp);
    }
    return ctrl_poses;
}

void CubicTrajectory::initializeCtrlPoses(PoseMap& poses,
                                          const ros::Time& t_beg,
                                          const ros::Time& t_end)
{
    // Compute the number of newly generated control poses
    const int num_cps = std::round((t_end - t_beg).toSec()/dt_knots_) + 3;
    // Fit control poses with the given poses
    std::vector<Sophus::SO3d> ctrl_poses = fitCtrlPoses(poses,t_beg_,num_cps);
    // Push back to this trajectory
    this->pushbackCtrlPoses(ctrl_poses);

    VLOG(4) << "New cubic trajectory initialized";
}

std::vector<Sophus::SO3d> CubicTrajectory::generateCtrlPoses(PoseMap& poses,
                                                             const ros::Time& t_beg,
                                                             const ros::Time& t_end)
{
    // Compute the number of newly generated control poses
    const int num_cps = std::round((t_end - t_beg).toSec()/dt_knots_) + 3;
    // Fit control poses with the given poses
    std::vector<Sophus::SO3d> ctrl_poses = fitCtrlPoses(poses,t_beg.toSec(),num_cps);
    return ctrl_poses;
}

std::vector<Sophus::SO3d> CubicTrajectory::generateCtrlPosesLong(PoseMap &poses,
                                                                 const ros::Time &t_beg,
                                                                 const ros::Time &t_end,
                                                                 const double sub_interval_length)
{
    const double timespan_poses = (t_end - t_beg).toSec();
    const size_t num_unit_invertal = std::floor(timespan_poses/sub_interval_length + 1e-6);

    // Generate control poses with front-end poses
    std::vector<Sophus::SO3d> ctrl_poses;
    for (size_t i = 0; i < num_unit_invertal; i++)
    {
        const ros::Time t_subset_beg = t_beg + ros::Duration(sub_interval_length) * i;
        const ros::Time t_subset_end = t_subset_beg + ros::Duration(sub_interval_length);

        // Get the pose subset for this sub-interval
        auto iter1 = poses.upper_bound(t_subset_beg);
        auto iter2 = poses.lower_bound(t_subset_end);
        PoseMap pose_subset;
        pose_subset.insert(iter1, iter2);
        std::vector<Sophus::SO3d> cp_subset = this->generateCtrlPoses(
                    pose_subset, t_subset_beg, t_subset_end);

        // If not the first sub-window, remove the first control pose
        if (i != 0)
        {
            cp_subset.erase(cp_subset.begin(), cp_subset.begin()+3);
        }

        // Add control poses into the whole control pose set
        for (auto cp : cp_subset)
        {
            ctrl_poses.push_back(cp);
        }
    }
    return ctrl_poses;
}


void CubicTrajectory::incrementalUpdate(const std::vector<Eigen::Vector3d>& drotv, const int& idx_beg)
{
    CHECK_EQ(idx_beg + drotv.size(), this->size());
    for (size_t i = idx_beg; i < this->size(); ++i)
    {
        // Use left perturbation to update control poses
        spline_.getKnot(i) = Sophus::SO3d::exp(drotv[i-idx_beg]) * spline_.getKnot(i);
    }
}

CubicTrajectory* CubicTrajectory::clone()
{
    std::vector<Sophus::SO3d> ctrl_poses;
    for (size_t i = 0; i < this->size(); i++)
    {
        ctrl_poses.push_back(spline_.getKnot(i));
    }
    CubicTrajectory* traj_clone = new CubicTrajectory(this->t_beg_, this->dt_knots_, ctrl_poses);
    return traj_clone;
}

CubicTrajectory* CubicTrajectory::cloneSegment(const size_t idx_cp_beg, const size_t idx_cp_end)
{
    CHECK_GT(idx_cp_end, idx_cp_beg);
    CHECK_GE(this->size(), idx_cp_end);
    std::vector<Sophus::SO3d> ctrl_poses;
    for (size_t i = idx_cp_beg; i < idx_cp_end; i++)
    {
        ctrl_poses.push_back(spline_.getKnot(i));
    }
    const double t_traj_seg_beg = this->t_beg_ + idx_cp_beg * this->dt_knots_;
    CubicTrajectory* traj_seg = new CubicTrajectory(t_traj_seg_beg, this->dt_knots_, ctrl_poses);
    return traj_seg;
}

void CubicTrajectory::replaceWith(Trajectory* traj_src, const size_t num_cp_copy,
                                  const size_t idx_cp_beg_src, const size_t idx_cp_beg_dst)
{
    CHECK_GE(traj_src->size(), idx_cp_beg_src + num_cp_copy);
    CHECK_GE(this->size(), idx_cp_beg_dst + num_cp_copy);

    // Replace the control poses of this trajectory with the ones from traj_src
    for (size_t i = 0; i < num_cp_copy; i++)
    {
        spline_.getKnot(idx_cp_beg_dst+i) = traj_src->getControlPose(idx_cp_beg_src+i);
    }
}

CubicTrajectory* CubicTrajectory::CopyAndIncrementalUpdate(const std::vector<Eigen::Vector3d>& drotv,
                                                           const int& idx_traj_beg,
                                                           const int& idx_opt_beg)
{
    CHECK_EQ(idx_opt_beg + drotv.size(), this->size());
    CHECK_GE(idx_opt_beg, idx_traj_beg);

    // Copy current trajectory to the temporary trajectory (only the part after idx_beg)
    std::vector<Sophus::SO3d> ctrl_poses_traj_temp;
    for (size_t i = idx_traj_beg; i < this->size(); i++)
    {
        ctrl_poses_traj_temp.push_back(spline_.getKnot(i));
    }
    const double t_traj_temp_beg = this->t_beg_ + idx_traj_beg * this->dt_knots_;
    CubicTrajectory* traj_temp = new CubicTrajectory(t_traj_temp_beg, this->dt_knots_, ctrl_poses_traj_temp);

    CHECK_EQ(drotv.size()+idx_opt_beg-idx_traj_beg, traj_temp->size());

    // Incrementally update this created trajectory
    traj_temp->incrementalUpdate(drotv, idx_opt_beg-idx_traj_beg);
    return traj_temp;
}
