#include "emba/model.h"
#include "utils/so3_funcs.h"

#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <unsupported/Eigen/SparseExtra>
#include "utils/eigen_utils.h"

#include <map>
#include <unordered_map>
#include <algorithm>

#include <glog/logging.h>

#define ENABLE_CHRONO

namespace EMBA {

/************* Base ***************/
void Model::updateTraj(Trajectory *traj_ptr, const VecXd& x1)
{
    // Use the solved perturbation (x1) to update the trajectory
    int num_drotv = x1.size()/3;
    CHECK_EQ(traj_ptr->size(), num_drotv);
    std::vector<Eigen::Vector3d> drotv;
    for (int i = 0; i < num_drotv; i++)
    {
        const Eigen::Vector3d drotv_i(x1[3*i], x1[3*i+1], x1[3*i+2]);
        drotv.push_back(drotv_i);
    }

    // Update control poses incrementally with left purturbation
    traj_ptr->incrementalUpdate(drotv, 0);
}

void Model::updateTraj(Trajectory* traj_ptr, const VecXd& x1, const size_t idx_cp_beg)
{
    // Use the solved perturbation (x1) to update the trajectory
    int num_drotv = x1.size()/3;
    CHECK_EQ(traj_ptr->size()-idx_cp_beg, num_drotv);

    std::vector<Eigen::Vector3d> drotv;
    for (int i = 0; i < num_drotv; i++)
    {
        const Eigen::Vector3d drotv_i(x1[3*i], x1[3*i+1], x1[3*i+2]);
        drotv.push_back(drotv_i);
    }

    // Update control poses incrementally with left purturbation, with the first control pose fixed
    traj_ptr->incrementalUpdate(drotv, 1);
}

/************* LEGM ***************/
LEGM::LEGM(const sensor_msgs::CameraInfo& camera_info_msg, double C_th,
           int pano_width, int pano_height)
{
    // Get the contrast threshold
    C_th_ = C_th;

    // Initialize the event warper
    event_warper_ptr_ = new dvs::EventWarper();
    event_warper_ptr_->initialize(camera_info_msg, pano_width, pano_height);

    // Initialize the event map
    const int sensor_width = camera_info_msg.width;
    const int sensor_height = camera_info_msg.height;
    event_map_ = EventMap<State_LEGM>(sensor_width, sensor_height);
}

VecXd LEGM::evaluateDataError(Trajectory *traj_ptr, const cv::Mat& Gx, const cv::Mat& Gy,
                              const EventPacket &events, bool eval_deriv, cv::Mat& num_ev_map)
{
    /** Data preparation **/
    // Process events in batch for speed-up
    size_t num_events = events.size();
    size_t event_batch_size = 100;
    size_t num_batches = std::ceil(num_events/event_batch_size);

    // Clear the event map
    event_map_.clear();

    // Clear the event number map
    num_ev_map.setTo(0);

    // Compute the second order gradient map via Sobel
    cv::Mat Gxx, Gxy, Gyx, Gyy; // Note: here Gxy and Gyx may not be same
    cv::Sobel(Gx, Gxx, CV_64FC1, 1, 0);
    cv::Sobel(Gx, Gxy, CV_64FC1, 0, 1);
    cv::Sobel(Gy, Gyx, CV_64FC1, 1, 0);
    cv::Sobel(Gy, Gyy, CV_64FC1, 0, 1);
    Gxx = 0.125 * Gxx;
    Gxy = 0.125 * Gxy;
    Gyx = 0.125 * Gyx;
    Gyy = 0.125 * Gyy;
    Gxy = 0.5 * (Gxy + Gyx); // Take the average of Gxy and Gyx !!!

    /** Process events **/
    // Loop through all event batches
    // ------------------------------------------------------------------------------------
    for (size_t i = 0; i < num_batches; i++)
    {
        // Get event batch
        size_t event_bgn_idx = event_batch_size * i;     // Head index
        size_t event_end_idx = event_batch_size * (i+1); // Tail index

        // If the index exceeds the total number of events (the last batch)
        if (event_end_idx > num_events)
        {
            event_end_idx = num_events;
            event_batch_size = event_end_idx - event_bgn_idx;
        }

        // Get the timestamp of the current event batch (mid point)
        const ros::Time t_batch_bgn = events.at(event_bgn_idx).ts;
        const ros::Time t_batch_end = events.at(event_end_idx-1).ts;
        ros::Duration timespan = t_batch_end - t_batch_bgn;
        ros::Time t_batch = t_batch_bgn + timespan * 0.5;

        // Query the pose at the middle point from the camera trajectory,
        // and compute the correspoding Jacobian
        // Jacobian: perturbation on Rot(t) w.r.t. involved control poses {cp_idx, cp_idx+1}
        cv::Mat ddrot_ddrot_cp; // 3X6
        int cp_idx;             // The index of the first involved control pose
        Sophus::SO3d rot;       // Rot(t_query): the rotation at t_query
        if (eval_deriv)
        {
            // Query the camera pose with the derivatives
            rot = traj_ptr->evaluate(t_batch, &cp_idx, &ddrot_ddrot_cp);
        }
        else
        {
            // Query the camera pose without the derivatives
            rot = traj_ptr->evaluate(t_batch, nullptr, nullptr);
        }

        // Loop through all events in this batch
        // ------------------------------------------------------------------------------------
        for (size_t k = event_bgn_idx; k < event_end_idx; k++)
        {
            // Warp events on to the pano map
            const int ev_x = events.at(k).x;
            const int ev_y = events.at(k).y;
            cv::Point2i pt_in(ev_x, ev_y);
            cv::Point2d pm;

            // Record the state for this event
            State_LEGM ev_state;

            // Jacobian: the location of this warped event w.r.t. the perturbation on Rot(t)
            cv::Matx23d dpm_ddrot; // 2X3
            if (eval_deriv)
            {
                pm = event_warper_ptr_->warpEventToMap(pt_in, rot, &dpm_ddrot);
                cv::Mat dpm_ddrot_cp = dpm_ddrot * ddrot_ddrot_cp;
                cv::cv2eigen(dpm_ddrot_cp, ev_state.dpm_ddrot_cp);
            }
            else
            {
                pm = event_warper_ptr_->warpEventToMap(pt_in, rot, nullptr);
            }

            // Save the state for this event
            ev_state.pol = events.at(k).polarity;
            ev_state.pm = pm;
            ev_state.cp_idx = cp_idx;

            // Add the state of this event into the event map
            event_map_.addEvent(ev_x, ev_y, ev_state);
        }
    }

    // After warping all the events, recover the error vector from the event map
    // Pre-allocate memory for ep, which needs to be resized (due to outliers)
    size_t inlier_count = 0;
    VecXd ep0 = VecXd::Zero(events.size());
    // Loop througn all the pixels of the event map
    for (int y = 0; y < event_map_.height(); y++)
    {
        for (int x = 0; x < event_map_.width(); x++)
        {
            // Get the states of all events at this sensor pixel (Use reference to avoid copying)
            std::vector<State_LEGM>& pix_ev_states = event_map_.at(x,y);
            // Loop through all the events occurred at this pixel, to form the error vector
            for (size_t k = 1; k < pix_ev_states.size(); k++) // Start from k = 1
            {
                // One measurement has two events: get the indexes of these two events
                // ev_prev: k-1, ev_curr: k
                State_LEGM& ev_curr_state = pix_ev_states.at(k);
                State_LEGM& ev_prev_state = pix_ev_states.at(k-1);

                // The displacement between these two consecutive warped events
                const ColVector2d dp(ev_curr_state.pm.x - ev_prev_state.pm.x,
                                     ev_curr_state.pm.y - ev_prev_state.pm.y);

                // Outlier rejection: if the displacement between these two warped events too small or too big,
                // this measurement (event pair) is outlier, which should be rejected
                const double dp_norm = dp.norm();
                if (dp_norm > 10)
                {
                    // Set the inlier index to be -1, to sign this event (as ev_curr) to be an outlier
                    ev_curr_state.inlier_idx = -1;
                    continue;
                }

                // The map gradient at the locations of the warped ev_curr
                // Select the closest pixel
                const int pm_x = static_cast<int>(std::round(ev_curr_state.pm.x));
                const int pm_y = static_cast<int>(std::round(ev_curr_state.pm.y));
                const cv::Point2i pm_int(pm_x, pm_y);
                // 1st-order spatial gradient of the map (dI/dpm)
                const RowVector2d Gpm(Gx.at<double>(pm_int),
                                      Gy.at<double>(pm_int));

                // Compute the predicted intensity change (contrast threshold)
                const double C_pred = Gpm * dp;
                // Compute the measured intensity change, using the polarity of ev_curr
                const double C_meas = 2*(ev_curr_state.pol-0.5)*C_th_;
                // Compute the error between predicted and actual measurement
                ep0(inlier_count) = C_meas - C_pred;

                // Update the inlier index of this event
                ev_curr_state.inlier_idx = inlier_count;
                inlier_count += 1;
                // Update the inlier event number at this pixel
                num_ev_map.at<int>(pm_int) += 1;

                // Add the pixel of the warped ev_curr and ev_prev into the list of active pixels
                if (eval_deriv)
                {
                    // 2nd-order spatial gradient of the map (dG/dpm)
                    Mat2d G2pm;
                    G2pm << Gxx.at<double>(pm_int), Gxy.at<double>(pm_int),
                            Gxy.at<double>(pm_int), Gyy.at<double>(pm_int); // 2X2

                    // Compute the derivatives needed by the normal equation and update the event map
                    RowVector2d temp = Gpm + dp.transpose()*G2pm; // 1X2
                    // Save these intermediate results into the event map
                    ev_curr_state.dp = dp;
                    ev_curr_state.Gpm = Gpm;
                    ev_curr_state.temp = temp;
                }
            }
        }
    }
    CHECK_GE(num_events, inlier_count);

//    // Print information
//    std::cout << "Number of inlier measurements: " << inlier_count << std::endl;
//    std::cout << "Number of total events: " << num_events << std::endl;
//    std::cout << "Percentage of inlier measurements: "
//              << 100.0*(float(inlier_count)/float(num_events)) << " %" << std::endl;

    // Shrink the zeros at the tail of the error vector (remove outlier measurements)
    const VecXd ep = ep0.head(inlier_count); // NOTE: there is not in-place operation in Eigen
    return ep;
}

VecXd LEGM::evaluateRegError(const cv::Mat &Gx, const cv::Mat &Gy)
{
    // Book memory
    const size_t num_total_pixels = Gx.rows * Gx.cols;
    VecXd ep = VecXd::Zero(2*num_total_pixels);

    // Fill in the error vector
    for (int y = 0; y < Gx.rows; y++)
    {
        for (int x = 0; x < Gx.cols; x++)
        {
            const size_t pm_idx = pxToIdx(x,y);
            ep(2*pm_idx)   = Gx.at<double>(cv::Point2i(x,y));
            ep(2*pm_idx+1) = Gy.at<double>(cv::Point2i(x,y));
        }
    }
    return ep;
}

double LEGM::evaluateRobustDataCost(const VecXd &ep, const std::string cost_type, const double a)
{
    if (cost_type == "cauchy")
    {
        // pho(u) = (1/(2*a))*ln(1+a*u^2)
        // Element-wise product x^2
        const VecXd temp1 = a * ep.cwiseProduct(ep);
        // Element-wise ln(x^2+1)
        const VecXd temp2 = temp1.array().log1p();
        // Sum up to obtain the cost
        const double cost = (0.5/a) * temp2.sum();
        return cost;
    }
    else // huber
    {
        double cost = 0;
        const double b = -0.5*a*a; // Linear part: intercept = -0.5*a^2
        const size_t dim_ep = ep.size();
        // Loop through all the error terms
        for (size_t k = 0; k < dim_ep; k++)
        {
            const double ep_k_abs = std::abs(ep(k));
            if (ep_k_abs < a)
            {
                // Quadratic interval: pho(u) = 0.5*u^2, when |u| < a
                cost += 0.5*ep_k_abs*ep_k_abs;
            }
            else
            {
                // Linear interval: pho(u) = a*|u|+b = a*(|u|-0.5*a), otherwise
                cost += a * ep_k_abs + b;
            }
        }
        return cost;
    }
}

void LEGM::formNormalEq(MatXd &A11, MatXd &A12, std::vector<Mat2d>& A22_blocks,
                        VecXd &b1, VecXd &b2, const VecXd &ep, const int num_ctrl_poses,
                        const cv::Mat& num_ev_map, const int thres_valid_pixel,
                        std::set<size_t>& active_pix_idxes, std::set<size_t>& inactive_pix_idxes)
{
    // Get the dimension of camera poses
    const int dim_ctrl_poses = 3*num_ctrl_poses;

    // Scan the event number map, and get the set of active/inactive pixels
    active_pix_idxes.clear();
    inactive_pix_idxes.clear();
    for (int y = 0; y < num_ev_map.rows; y++)
    {
        for (int x = 0; x < num_ev_map.cols; x++)
        {
            const size_t pm_idx = pxToIdx(x,y);
            // If this pixel has enough warped events
            if (num_ev_map.at<int>(y,x) >= thres_valid_pixel)
            {
                // Add to the set of valid pixels
                active_pix_idxes.insert(pm_idx);
            }
            else
            {
                // Add to the set of invalid pixels
                inactive_pix_idxes.insert(pm_idx);
            }
        }
    }

    // Check if "total pixels = valid + invalid"?
    const size_t num_active_pixels = active_pix_idxes.size();
    const size_t num_inactive_pixels = inactive_pix_idxes.size();
    const size_t num_total_pixels = num_ev_map.rows * num_ev_map.cols;
    CHECK_EQ(num_total_pixels, num_active_pixels+num_inactive_pixels);

//    const double active_pixel_percentage = 100.0 * double(num_active_pixels)/double(num_total_pixels);
//    std::cout << "[LEGM::formNormalEq] active_pixel_percentage = "
//              << active_pixel_percentage << " %" << std::endl;

    // Memory allocation
    A11 = MatXd::Zero(dim_ctrl_poses, dim_ctrl_poses);      // A11 is dense
    A12 = MatXd::Zero(dim_ctrl_poses, 2*num_active_pixels); // A12 is dense
    A22_blocks.clear();
    A22_blocks = std::vector<Mat2d>(num_active_pixels);     // A22 is 2X2 block diagonal
    b1 = VecXd::Zero(dim_ctrl_poses);
    b2 = VecXd::Zero(2*num_active_pixels);

    // Initialize all A22 blocks to be zeros
    for (auto it = A22_blocks.begin(); it != A22_blocks.end(); it++)
    {
        *it = Mat2d::Zero();
    }

    // Build a look-up table for active pixels <index on the pano map, index in the active pixel set>
    std::unordered_map<size_t, size_t> active_pix_map;
    size_t pix_count = 0;
    for (auto idx: active_pix_idxes)
    {
        active_pix_map.insert(std::pair<size_t, size_t>(idx, pix_count));
        pix_count += 1;
    }
    CHECK_EQ(active_pix_map.size(), active_pix_idxes.size());
    CHECK_EQ(pix_count, active_pix_idxes.size());

    // Loop through all pixels on the event map,
    // to form the normal equation in a cumulative way
    for (int y = 0; y < event_map_.height(); y++)
    {
        for (int x = 0; x < event_map_.width(); x++)
        {
            // Get the states of all events at this pixel
            std::vector<State_LEGM>& pix_ev_states = event_map_.at(x,y);
            // Loop through all the events occurred at this pixel, to form the normal equation
            for (size_t k = 1; k < pix_ev_states.size(); k++) // Start from k = 1
            {
                // One measurement has two events: get the indexes of these two events
                // ev_prev: k-1, ev_curr: k
                State_LEGM& ev_curr_state = pix_ev_states.at(k);
                // Check if this event is outlier
                if (ev_curr_state.inlier_idx < 0)
                {
                    continue;
                }
                State_LEGM& ev_prev_state = pix_ev_states.at(k-1);

                // The index of the pixel that contains the warped ev_curr
                // Select the closest pixel
                const int pm_curr_x = static_cast<int>(std::round(ev_curr_state.pm.x));
                const int pm_curr_y = static_cast<int>(std::round(ev_curr_state.pm.y));

                // Check if this event is warped to any valid pixel?
                // If not, skip this event
                if (num_ev_map.at<int>(pm_curr_y, pm_curr_x) < thres_valid_pixel)
                {
                    continue;
                }

                const size_t pm_curr_idx = pxToIdx(pm_curr_x, pm_curr_y);
                // Search for the index of the pixel in the set of active pixels
                const size_t pm_curr_active_idx = active_pix_map.find(pm_curr_idx)->second;

                // Accumulate the contribution of this event to the normal equation
                // ---------------------------------------------------------------------------
                // Measurement error caused by this event
                const double ep_k = ep(ev_curr_state.inlier_idx);

                // (1) Map part
                // ---------------------------------------------------------------------------
                // - Left hand side: A22
                const double dM_dGx = ev_curr_state.dp(0);
                const double dM_dGy = ev_curr_state.dp(1);

                const double A22_k_xx = dM_dGx*dM_dGx;
                const double A22_k_xy = dM_dGx*dM_dGy;
                const double A22_k_yy = dM_dGy*dM_dGy;
                Mat2d A22_k;
                A22_k << A22_k_xx, A22_k_xy,
                         A22_k_xy, A22_k_yy;
                A22_blocks.at(pm_curr_active_idx) += A22_k;

                // - Right hand side: b2
                b2(2*pm_curr_active_idx)   += dM_dGx*ep_k;
                b2(2*pm_curr_active_idx+1) += dM_dGy*ep_k;

                // (2) Pose part
                // ---------------------------------------------------------------------------
                // - Left hand side: A11
                // ---------------------------------------------------------------------------
                // Accumulate the contribution of this measurement in matrix A11

                // -- Contribution of ev_curr
                // The derivatives of the measurement model w.r.t. the 1st and 2nd involved control poses of ev_curr
                RowVector6d dM_ddrot_cp_curr = ev_curr_state.temp * ev_curr_state.dpm_ddrot_cp;

                // Position index of the (ev_curr) involved control poses in matrix A11
                // (linear spline, 2 control poses = 6 dimensions)
                const size_t start_dim_ev_curr = 3*ev_curr_state.cp_idx;
                A11.block(start_dim_ev_curr, start_dim_ev_curr, 6, 6) +=
                        dM_ddrot_cp_curr.transpose() * dM_ddrot_cp_curr;

                // -- Contribution of ev_prev
                // The derivatives of the measurement model w.r.t. the 1st and 2nd involved control poses of ev_prev
                RowVector6d dM_ddrot_cp_prev = -ev_curr_state.Gpm * ev_prev_state.dpm_ddrot_cp;

                // Position index of the (previous event) involved control poses in matrix A
                const size_t start_dim_ev_prev = 3*ev_prev_state.cp_idx;
                A11.block(start_dim_ev_prev, start_dim_ev_prev, 6, 6) +=
                        dM_ddrot_cp_prev.transpose() * dM_ddrot_cp_prev;

                // Contribution of the cross terms of ev_curr and ev_prev
                const Mat6d cross_block = dM_ddrot_cp_curr.transpose() * dM_ddrot_cp_prev;
                A11.block(start_dim_ev_curr, start_dim_ev_prev, 6, 6) += cross_block;
                A11.block(start_dim_ev_prev, start_dim_ev_curr, 6, 6) += cross_block.transpose();

                // - Right hand side: b1
                // ---------------------------------------------------------------------------
                // Accumulate the contribution of this measurement to vector b1
                // - Contribution of ev_curr
                b1.block(start_dim_ev_curr,0,6,1) += dM_ddrot_cp_curr.transpose() * ep_k;
                // - Contribution of ev_prev
                b1.block(start_dim_ev_prev,0,6,1) += dM_ddrot_cp_prev.transpose() * ep_k;

                // (3) Map-pose cross part: A12
                // ---------------------------------------------------------------------------
                // Accumulate the contribution of this measurement in matrix A12
                // - Contribution of ev_curr
                A12.block(start_dim_ev_curr, 2*pm_curr_active_idx,   6, 1) += dM_ddrot_cp_curr.transpose() * dM_dGx;
                A12.block(start_dim_ev_curr, 2*pm_curr_active_idx+1, 6, 1) += dM_ddrot_cp_curr.transpose() * dM_dGy;
                // - Contribution of ev_prev
                A12.block(start_dim_ev_prev, 2*pm_curr_active_idx,   6, 1) += dM_ddrot_cp_prev.transpose() * dM_dGx;
                A12.block(start_dim_ev_prev, 2*pm_curr_active_idx+1, 6, 1) += dM_ddrot_cp_prev.transpose() * dM_dGy;
            }
        }
    }
}

void LEGM::formNormalEqIRLS(MatXd& A11, MatXd& A12, std::vector<Mat2d>& A22_blocks,
                            VecXd& b1, VecXd& b2, const VecXd& ep, const int num_ctrl_poses,
                            const cv::Mat& num_ev_map, const int thres_valid_pixel,
                            std::set<size_t>& active_pix_idxes, std::set<size_t>& inactive_pix_idxes,
                            const std::string cost_type, const double a)
{
    // Get the dimension of camera poses
    const int dim_ctrl_poses = 3*num_ctrl_poses;

    // Scan the event number map, and get the set of active/inactive pixels
    active_pix_idxes.clear();
    inactive_pix_idxes.clear();
    for (int y = 0; y < num_ev_map.rows; y++)
    {
        for (int x = 0; x < num_ev_map.cols; x++)
        {
            const size_t pm_idx = pxToIdx(x,y);
            // If this pixel has enough warped events
            if (num_ev_map.at<int>(y,x) >= thres_valid_pixel)
            {
                // Add to the set of valid pixels
                active_pix_idxes.insert(pm_idx);
            }
            else
            {
                // Add to the set of invalid pixels
                inactive_pix_idxes.insert(pm_idx);
            }
        }
    }

    // Check if "total pixels = valid + invalid"?
    const size_t num_active_pixels = active_pix_idxes.size();
    const size_t num_inactive_pixels = inactive_pix_idxes.size();
    const size_t num_total_pixels = num_ev_map.rows * num_ev_map.cols;
    CHECK_EQ(num_total_pixels, num_active_pixels+num_inactive_pixels);



    // Memory allocation
    A11 = MatXd::Zero(dim_ctrl_poses, dim_ctrl_poses);      // A11 is dense
    A12 = MatXd::Zero(dim_ctrl_poses, 2*num_active_pixels); // A12 is dense
    A22_blocks.clear();
    A22_blocks = std::vector<Mat2d>(num_active_pixels);     // A22 is 2X2 block diagonal
    b1 = VecXd::Zero(dim_ctrl_poses);
    b2 = VecXd::Zero(2*num_active_pixels);

    // Initialize all A22 blocks to be zeros
    for (auto it = A22_blocks.begin(); it != A22_blocks.end(); it++)
    {
        *it = Mat2d::Zero();
    }

    // Build a look-up table for active pixels <index on the pano map, index in the active pixel set>
    std::unordered_map<size_t, size_t> active_pix_map;
    size_t pix_count = 0;
    for (auto idx: active_pix_idxes)
    {
        active_pix_map.insert(std::pair<size_t, size_t>(idx, pix_count));
        pix_count += 1;
    }
    CHECK_EQ(active_pix_map.size(), active_pix_idxes.size());
    CHECK_EQ(pix_count, active_pix_idxes.size());

    // Loop through all pixels on the event map,
    // to form the normal equation in a cumulative way
    for (int y = 0; y < event_map_.height(); y++)
    {
        for (int x = 0; x < event_map_.width(); x++)
        {
            // Get the states of all events at this pixel
            std::vector<State_LEGM>& pix_ev_states = event_map_.at(x,y);
            // Loop through all the events occurred at this pixel, to form the normal equation
            for (size_t k = 1; k < pix_ev_states.size(); k++) // Start from k = 1
            {
                // One measurement has two events: get the indexes of these two events
                // ev_prev: k-1, ev_curr: k
                State_LEGM& ev_curr_state = pix_ev_states.at(k);
                // Check if this event is outlier
                if (ev_curr_state.inlier_idx < 0)
                {
                    continue;
                }
                State_LEGM& ev_prev_state = pix_ev_states.at(k-1);

                // The index of the pixel that contains the warped ev_curr
                // Select the closest pixel
                const int pm_curr_x = static_cast<int>(std::round(ev_curr_state.pm.x));
                const int pm_curr_y = static_cast<int>(std::round(ev_curr_state.pm.y));

                // Check if this event is warped to any valid pixel?
                // If not, skip this event
                if (num_ev_map.at<int>(pm_curr_y, pm_curr_x) < thres_valid_pixel)
                {
                    continue;
                }

                // Search for the index of the pixel in the set of active pixels
                const size_t pm_curr_idx = pxToIdx(pm_curr_x, pm_curr_y);
                const size_t pm_curr_active_idx = active_pix_map.find(pm_curr_idx)->second;

                // Accumulate the contribution of this event to the normal equation
                // ---------------------------------------------------------------------------
                // Measurement error caused by this event
                const double ep_k = ep(ev_curr_state.inlier_idx);

                // IRLS: compute the weight (inverse covariance) of this event (error term)
                double Yi_inv; // New weight (inverse covariance)
                if (cost_type == "cauchy")
                {
                    Yi_inv = 1.0/(1.0 + a*ep_k*ep_k);
                }
                else
                {
                    // Huber
                    const double ep_k_abs = std::abs(ep_k);
                    if (ep_k_abs < a)
                    {
                        Yi_inv = 1.0;
                    }
                    else
                    {
                        Yi_inv = a/ep_k_abs;
                    }
                }
                const double ep_k_reweighted = Yi_inv * ep_k;

                // (1) Map part
                // ---------------------------------------------------------------------------
                // - Left hand side: A22
                const double dM_dGx = ev_curr_state.dp(0);
                const double dM_dGy = ev_curr_state.dp(1);

                const double A22_k_xx = dM_dGx*dM_dGx;
                const double A22_k_xy = dM_dGx*dM_dGy;
                const double A22_k_yy = dM_dGy*dM_dGy;
                Mat2d A22_k;
                A22_k << A22_k_xx, A22_k_xy,
                         A22_k_xy, A22_k_yy;
                A22_blocks.at(pm_curr_active_idx) += Yi_inv * A22_k;

                // - Right hand side: b2
                b2(2*pm_curr_active_idx)   += dM_dGx * ep_k_reweighted;
                b2(2*pm_curr_active_idx+1) += dM_dGy * ep_k_reweighted;

                // (2) Pose part
                // ---------------------------------------------------------------------------
                // - Left hand side: A11
                // ---------------------------------------------------------------------------
                // Accumulate the contribution of this measurement in matrix A11
                // -- Contribution of ev_curr
                // The derivatives of the measurement model w.r.t. the 1st and 2nd involved control poses of ev_curr
                RowVector6d dM_ddrot_cp_curr = ev_curr_state.temp * ev_curr_state.dpm_ddrot_cp;

                // Position index of the (ev_curr) involved control poses in matrix A11
                // (linear spline, 2 control poses = 6 dimensions)
                const size_t start_dim_ev_curr = 3*ev_curr_state.cp_idx;
                A11.block(start_dim_ev_curr, start_dim_ev_curr, 6, 6) +=
                        Yi_inv * dM_ddrot_cp_curr.transpose() * dM_ddrot_cp_curr;

                // -- Contribution of ev_prev
                // The derivatives of the measurement model w.r.t. the 1st and 2nd involved control poses of ev_prev
                RowVector6d dM_ddrot_cp_prev = -ev_curr_state.Gpm * ev_prev_state.dpm_ddrot_cp;

                // Position index of the (previous event) involved control poses in matrix A
                const size_t start_dim_ev_prev = 3*ev_prev_state.cp_idx;
                A11.block(start_dim_ev_prev, start_dim_ev_prev, 6, 6) +=
                        Yi_inv * dM_ddrot_cp_prev.transpose() * dM_ddrot_cp_prev;

                // Contribution of the cross terms of ev_curr and ev_prev
                const Mat6d cross_block = Yi_inv * dM_ddrot_cp_curr.transpose() * dM_ddrot_cp_prev;
                A11.block(start_dim_ev_curr, start_dim_ev_prev, 6, 6) += cross_block;
                A11.block(start_dim_ev_prev, start_dim_ev_curr, 6, 6) += cross_block.transpose();

                // - Right hand side: b1
                // ---------------------------------------------------------------------------
                // Accumulate the contribution of this measurement to vector b1
                // - Contribution of ev_curr
                b1.block(start_dim_ev_curr,0,6,1) += dM_ddrot_cp_curr.transpose() * ep_k_reweighted;
                // - Contribution of ev_prev
                b1.block(start_dim_ev_prev,0,6,1) += dM_ddrot_cp_prev.transpose() * ep_k_reweighted;

                // (3) Map-pose cross part: A12
                // ---------------------------------------------------------------------------
                // Accumulate the contribution of this measurement in matrix A12
                // - Contribution of ev_curr
                A12.block(start_dim_ev_curr, 2*pm_curr_active_idx,   6, 1) += Yi_inv * dM_ddrot_cp_curr.transpose() * dM_dGx;
                A12.block(start_dim_ev_curr, 2*pm_curr_active_idx+1, 6, 1) += Yi_inv * dM_ddrot_cp_curr.transpose() * dM_dGy;
                // - Contribution of ev_prev
                A12.block(start_dim_ev_prev, 2*pm_curr_active_idx,   6, 1) += Yi_inv * dM_ddrot_cp_prev.transpose() * dM_dGx;
                A12.block(start_dim_ev_prev, 2*pm_curr_active_idx+1, 6, 1) += Yi_inv * dM_ddrot_cp_prev.transpose() * dM_dGy;
            }
        }
    }
}

void LEGM::applyL2Reg(std::vector<Mat2d> &A22_blocks, VecXd &b2,
                      const std::set<size_t> &active_pix_idxes,
                      const double alpha, const cv::Mat &Gx, const cv::Mat &Gy)
{
    // Get the number of active pixels
    const size_t num_active_pix = active_pix_idxes.size();

    // 1. Modify the LHS matrix A22: A22 = A22 + alpha * I
    Mat2d I; // 2x2 identity matrix
    I.setIdentity();
    const Mat2d alpha_I = alpha * I;

    for (size_t i = 0; i < num_active_pix; ++i)
    {
        A22_blocks.at(i) += alpha_I;
    }

    // 2. Modify the RHS vector b2: b2 = b2 - alpha * x_op
    VecXd x_op(2*num_active_pix);
    x_op.setZero();
    size_t j = 0;
    for (auto p:active_pix_idxes)
    {
        // Convert to the pano map coordinate
        const cv::Point2i pm = idxToPx(p);
        x_op(2*j)   = Gx.at<double>(pm);
        x_op(2*j+1) = Gy.at<double>(pm);
        j += 1;
    }
    b2 -= alpha * x_op;
}

void LEGM::solveNormalEq(const MatXd &A11, const MatXd &A12, const std::vector<Mat2d> &A22_blocks,
                         const VecXd &b1, const VecXd &b2, const double lambda, VecXd &x1, VecXd &x2)
{
    const size_t dim_map = b2.size();

    // Apply LM modification to A11
    // Option 1: A11m = A11 + lambda * diag(A11)
    const VecXd diag_A11 = A11.diagonal();
    MatXd D11 = eigen_utils::diagMat(diag_A11);
    const MatXd A11m = A11 + lambda * D11;

    // The data structure that maintains the non-zero elements in the inverse of A22m_inv
    // Here we maintain the diagnoal and lower triangular part sperately.
    VecXd A22m_inv_diag_nonzeros = VecXd::Zero(dim_map); // All elements on the diagonal are non-zero
    // Maintain non-diagonal (lower only, due to the symmertic pattern) non-zeros elements
    std::unordered_map<size_t, double> A22m_inv_lower_nonzeros_map;

    // 2X2 identity matrix
    Mat2d I22_i;
    I22_i.setIdentity();

    // Exploit the block diagonal structure of A22, to compute inv(A22) and W
    for (size_t i = 0; i < A22_blocks.size(); i++)
    {
        // Apply LM modification to each block
        const Mat2d A22_i = A22_blocks.at(i);
        // A22m = A22 + lambda * diag(A22)
        const Mat2d A22m_i = A22_i + lambda * eigen_utils::diagMat2d(A22_i.diagonal());
        // Invert A22m in a block-by-block manner
        const Mat2d A22m_i_inv = A22m_i.inverse();

        // Insert the non-zero elements of A22m_i_inv into the non-zeros of A22m_inv
        A22m_inv_diag_nonzeros(2*i)   = A22m_i_inv(0,0);
        A22m_inv_diag_nonzeros(2*i+1) = A22m_i_inv(1,1);
        // Insert the non-diagonal (lower triangular) non-zero elements
        // For the location of this nonzero, we only need to save the col index: 2*i, then the row index is 2*i+1
        // i.e., the location of this non-zero entry is (2*i+1, 2*i)
        A22m_inv_lower_nonzeros_map.insert(std::pair<size_t,double>(2*i, A22m_i_inv(1,0)));
    }

    // Recover A22m_inv from its non-zero elements
    // (1) Create a diagonal matrix with A22m_inv's diagonal
    SpMat D22 = eigen_utils::diagSpMat(A22m_inv_diag_nonzeros);
    // (2) Create a matrix with A22's non-diagonal non-zero elements
    std::vector<Triplet> A22m_inv_triangular_nonzeros_vector;
    A22m_inv_triangular_nonzeros_vector.reserve(2*A22m_inv_lower_nonzeros_map.size());
    for (auto it = A22m_inv_lower_nonzeros_map.begin(); it != A22m_inv_lower_nonzeros_map.end(); it++)
    {
        const size_t col = it->first;
        const size_t row = it->first+1;
        A22m_inv_triangular_nonzeros_vector.push_back(Triplet(row,col,it->second));
        A22m_inv_triangular_nonzeros_vector.push_back(Triplet(col,row,it->second));
    }
    SpMat A22m_inv = SpMat(dim_map, dim_map);
    A22m_inv.setFromTriplets(A22m_inv_triangular_nonzeros_vector.begin(),
                             A22m_inv_triangular_nonzeros_vector.end());
    // Add A22m_inv's diagonal to recover the full A22
    A22m_inv += D22;
    // Convert A22 to compressed storage, to save memory
    A22m_inv.makeCompressed();

    // Solve the normal equations by means of Schur complement
    // W = A12 * inv(A22)
    const MatXd W = A12 * A22m_inv;
    // (1) Compute Schur complement: S = A11 - W * A12'
    const MatXd S = A11m - W * A12.transpose();

    // (2) Solver for x1: x1 = S\(b1-W*b2)
    x1 = S.ldlt().solve(b1-W*b2); // TODO: too see if Schur is S.P.D.?
    // (3) Solve for x2
    x2 = A22m_inv * (b2 - A12.transpose()*x1);
}

std::pair<int, double> LEGM::solveNormalEqCG(const MatXd& A11, const MatXd& A12,
                                             const std::vector<Mat2d>& A22_blocks,
                                             const VecXd& b1, const VecXd& b2,
                                             const double lambda, VecXd& x1, VecXd& x2)
{
    const size_t dim_poses = b1.size();
    const size_t dim_map = b2.size();

    // Recover the A22 from its non-zero blocks
    SpMat A22 = recoverA22FromBlocks(A22_blocks);

    // Apply LM modification to A11
    const VecXd diag_A11 = A11.diagonal();
    MatXd D11 = eigen_utils::diagMat(diag_A11);
    const MatXd A11m = A11 + lambda * D11;

    // Apply LM modification to A22
    const VecXd diag_A22 = A22.diagonal();
    SpMat D22 = eigen_utils::diagSpMat(diag_A22);
    SpMat A22m = A22 + lambda * D22;

    /// Solve by the Conjugate Gradient method (iterative)
    // Matrix concatenation
    const size_t dim_total = dim_poses + dim_map;
    VecXd b(dim_total);
    b << b1, b2; // b = [b1; b2]
    SpMat At, Ab, A;
    // If dim == 1, then C = [A;B]; If dim == 2 then C = [A B].
    eigen_utils::catSpMat(2, A11m.sparseView(), A12.sparseView(), At); // At = [A11 A12]
    eigen_utils::catSpMat(2, A12.sparseView().transpose(), A22m, Ab);  // Ab = [A12' A22]
    eigen_utils::catSpMat(1, At, Ab, A); //  A = [At; Ab]

    /* Solve using the built-in Eigen solver */
    Eigen::ConjugateGradient<SpMat, Eigen::Lower|Eigen::Upper> cg;
    const int max_iter = 100;
    const double tol = 1e-6;
    cg.setMaxIterations(max_iter);
    cg.setTolerance(tol);
    cg.compute(A);
    VecXd x = cg.solve(b);

    // Split x into x1 and x2
    x1 = x.head(dim_poses);
    x2 = x.tail(dim_map);

    return std::pair<int, double>(cg.iterations(), cg.error());
}

SpMat LEGM::recoverA22FromBlocks(const std::vector<Mat2d> &A22_blocks)
{
    // Recover A22 from the blocks
    std::vector<Triplet> A22_nonzeros_vector;
    const size_t num_active_pixels = A22_blocks.size();
    A22_nonzeros_vector.reserve(4*num_active_pixels);
    for (size_t i = 0; i < A22_blocks.size(); i++)
    {
        // Two non-zero elements on the diagonal
        A22_nonzeros_vector.push_back(Triplet(2*i,  2*i,  A22_blocks.at(i)(0,0)));
        A22_nonzeros_vector.push_back(Triplet(2*i+1,2*i+1,A22_blocks.at(i)(1,1)));
        // Two non-zero elements at the triangular parts
        A22_nonzeros_vector.push_back(Triplet(2*i+1,2*i  ,A22_blocks.at(i)(1,0)));
        A22_nonzeros_vector.push_back(Triplet(2*i,  2*i+1,A22_blocks.at(i)(0,1)));
    }
    SpMat A22 = SpMat(2*num_active_pixels,2*num_active_pixels);
    A22.setFromTriplets(A22_nonzeros_vector.begin(), A22_nonzeros_vector.end());
    A22.makeCompressed();
    return A22;
}

void LEGM::updateMap(cv::Mat& Gx, cv::Mat& Gy, const VecXd& x2, const double damping_factor,
                     const std::set<size_t>& active_pix_idxes, const std::set<size_t>& inactive_pix_idxes)
{
    // Loop through all active pixels and update them
    /// 1. Update active pixels
    size_t i = 0;
    for (auto p:active_pix_idxes)
    {
        // Convert to the pano map coordinate
        const cv::Point2i pm = idxToPx(p);
        // Update the gradient map
        Gx.at<double>(pm) += damping_factor * x2(2*i);
        Gy.at<double>(pm) += damping_factor * x2(2*i+1);
        // Switch to the next valid pixel
        i++;
    }
    CHECK_EQ(i, active_pix_idxes.size());

    /// 2. Update inactive pixels (set Gx and Gy to zero)
    // Compute the total number of pixels: H x W
    const size_t num_total_pix = event_warper_ptr_->panoMapWidth()
                                 * event_warper_ptr_->panoMapHeight();
    // Get the number of valid pixels
    const size_t num_active_pix = active_pix_idxes.size();
    // Compute the number of invalid pixels
    const size_t num_inactive_pix = inactive_pix_idxes.size();
    CHECK_EQ(num_total_pix, num_active_pix+num_inactive_pix);

    size_t j = 0;
    for (auto p:inactive_pix_idxes)
    {
        // Convert to the pano map coordinate
        const cv::Point2i pm = idxToPx(p);
        // Update the gradient map
        Gx.at<double>(pm) = 0.0;
        Gy.at<double>(pm) = 0.0;
        // Switch to the next inactive pixel
        j++;
    }
    CHECK_EQ(j, inactive_pix_idxes.size());
}
}
