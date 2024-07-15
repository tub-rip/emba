#include "utils/image_utils.h"
#include "emba/emba.h"
#include <filesystem>
#include <unsupported/Eigen/SparseExtra>

#define ENABLE_CHRONO

namespace EMBA {

// Using Levenberg-Marquart (L-M) method to solve BA
VecXd EMBA::solveTimeWindow(Trajectory* &traj_ptr, cv::Mat& Gx, cv::Mat& Gy,
                            const EventPacket& event_subset)
{
    /// (1) Initialize
    double lambda = 1e-3;        // Exploration/Exploitation parameter
    double lambda_max = 1e3;     // Maximum value of the exploration parameter
    double lambda_min = 1e-300;  // Minimum value of the exploration parameter

    double cost_min_old = 1e99;       // Last minimum cost
    double cost_new = cost_min_old;   // Cost at new point
    double cost_min = cost_min_old;   // Minimum cost so far

    int iter = 0;                     // Iteration counter
    int count_tol_fun_sat = 0;        // Counter of consecutive times tolerance has been satisfied
    bool cost_has_decreased = true;   // Flag to indicate if cost has decreased in current iteration

    // Measurement errors
    VecXd ep_data, ep_data_new, ep_reg, ep_reg_new, e_data_min, e_reg_min;
    double cost_data, cost_reg, cost_data_new, cost_reg_new;
    // Normal equation
    MatXd A11, A12;                 // Left-hand-side
    std::vector<Mat2d> A22_blocks;  // A22 is 2X2 block diagonal
    VecXd b1, b2;                   // Right-hand-side
    VecXd x1, x2;                   // Solution

    // L2 regularization on the gradient map
    const double alpha = BA_config.alpha;

    // Number of control poses
    const int num_ctrl_poses = traj_ptr->size();

    // The map to save the number of warped events at each pixel
    cv::Mat num_ev_map, num_ev_map_new;
    num_ev_map = cv::Mat::zeros(Gx.rows, Gx.cols, CV_32SC1);
    num_ev_map_new = cv::Mat::zeros(Gx.rows, Gx.cols, CV_32SC1);

    // The sets of active/inactive pixels
    std::set<size_t> active_pix_idxes, inactive_pix_idxes;

    // Data visualization & recording
    std::stringstream ss_window_id;
    ss_window_id << "win_" << std::setfill('0') << std::setw(4) << count_window_ << "_";
    std::string str_win_id = ss_window_id.str();

    if (record_data_)
    {
        iteration_writer_ << "window #" << count_window_ << std::endl;
        iteration_writer_ << "---------------------------------------------------------" << std::endl;
    }

    // Start iterations
    // --------------------------------------------------------------------------
    while(iter <= LM_params.max_num_iter && cost_min > 1e-16
          && lambda <= lambda_max && lambda >= lambda_min)
    {
        if (cost_has_decreased)
        {
            /// (2) Use the results from the previous iteration because cost has
            /// decreased. Do not evaluate again model function at new point
            if (iter == 0)
            {
                // In the first iteration, compute the estimated measurement
                // vector, the Jacobian matrix, the error vector
                // 1. Evaluate the error caused by the event data
                ep_data = model_ptr_->evaluateDataError(traj_ptr, Gx, Gy, event_subset,
                                                        true, num_ev_map);
                // 2. Evaluate the error caused by the L2 regularization on the gradient map
                ep_reg = model_ptr_->evaluateRegError(Gx, Gy);

                // Initialize output arguments: minimum cost and the measurement
                // vector at the minimum.
                if (BA_config.use_IRLS)
                {
                    cost_data = model_ptr_->evaluateRobustDataCost(ep_data, BA_config.cost_type, BA_config.eta);
                }
                else
                {
                    cost_data = 0.5 * ep_data.dot(ep_data);
                }
                cost_reg = alpha * 0.5 * ep_reg.dot(ep_reg);
                cost_min = cost_data + cost_reg;
            }
            else
            {
                // Do not evaluate again model function at new point
                ep_data = ep_data_new;
                ep_reg = ep_reg_new;

                // NOTE: we do not need to update the event map here, because it is only used
                //       when the cost decreases (then the event map is already updated
                //       inside the model)
                num_ev_map_new.copyTo(num_ev_map);
            }

#ifdef ENABLE_CHRONO
                static unsigned long count_formEqs = 0; // Number of times of solving normal equations
                static long double t_total_formEqs = 0; // [s]
                std::chrono::high_resolution_clock::time_point t1_formEqs =
                        std::chrono::high_resolution_clock::now();
#endif

            /// (3) Form the normal equation
            // Update the normal equation only when the states are updated
            if (BA_config.use_IRLS)
            {
                model_ptr_->formNormalEqIRLS(A11, A12, A22_blocks, b1, b2, ep_data, num_ctrl_poses,
                                             num_ev_map, BA_config.thres_valid_pixel,
                                             active_pix_idxes, inactive_pix_idxes,
                                             BA_config.cost_type, BA_config.eta);
            }
            else
            {
                model_ptr_->formNormalEq(A11, A12, A22_blocks, b1, b2, ep_data, num_ctrl_poses,
                                         num_ev_map, BA_config.thres_valid_pixel,
                                         active_pix_idxes, inactive_pix_idxes);
            }


            // Apply L2 regularization on the gradient map
            model_ptr_->applyL2Reg(A22_blocks, b2, active_pix_idxes, alpha, Gx, Gy);


#ifdef ENABLE_CHRONO
            std::chrono::high_resolution_clock::time_point t2_formEqs =
                    std::chrono::high_resolution_clock::now();
            auto duration_formEqs =
                    std::chrono::duration_cast<std::chrono::milliseconds>(t2_formEqs - t1_formEqs).count();
            count_formEqs += 1;
            t_total_formEqs += duration_formEqs;
            double sec_total_formEqs = t_total_formEqs/1e3;
            double sec_average_formEqs = sec_total_formEqs/count_formEqs;

            VLOG(1) << "[NormalEqs Forming] count_formEqs: " << count_formEqs;
            VLOG(1) << "[NormalEqs Forming] sec_total_formEqs: " << sec_total_formEqs;
            VLOG(1) << "[NormalEqs Forming] sec_average_formEqs: " << sec_average_formEqs;

            std::ofstream runtime_formEqs(result_dir_+"/final_results/runtime_formEqs.txt", std::ios::app);
            runtime_formEqs << "iter #" << iter << " count_formEqs = " << count_formEqs
                            << " sec_total_formEqs = " << sec_total_formEqs
                            << " sec_average_formEqs = " << sec_average_formEqs << std::endl;
#endif

            // For the first time window, we should fix the first control pose
            // A convenient way is to remove the rows and columns
            // that are related to the first control pose from A11, A12 and b1
            if (first_time_window_)
            {
                const size_t dim_pose_left = 3*(num_ctrl_poses-1);                // It seems that there is no in-place copy in Eigen, so we need these "bridge" variables
                MatXd A11_1st = A11.block(3, 3, dim_pose_left, dim_pose_left);
                MatXd A12_1st = A12.block(3, 0, dim_pose_left, A12.cols());
                VecXd b1_1st = b1.tail(dim_pose_left);
                A11 = A11_1st;
                A12 = A12_1st;
                b1 = b1_1st;
            }
        }

        // Display iteration progress: iteration counter, exploration
        // parameter (in log scale), minimum cost and cost at new point
        VLOG(0) << "iter #" << iter << ":  log10(lambda) = " << std::log10(lambda)
                << "  cost_min^2 = " << cost_min << "  cost_new^2 = " << cost_new;

        if (record_data_)
        {
            saveEvoData(Gx, Gy, str_win_id, iter);
            iteration_writer_ << "iter #" << iter << ":  log10(lambda) = " << std::log10(lambda)
                              << "  cost_min^2 = " << cost_min << "  cost_new^2 = " << cost_new
                              << "  cost_data = " << cost_data << "  cost_reg = " << cost_reg << std::endl;
        }

#ifdef ENABLE_CHRONO
        static unsigned long count_solveEqs = 0; // Number of times of solving normal equations
        static long double t_total_solveEqs = 0; // [s]
        std::chrono::high_resolution_clock::time_point t1_solveEqs =
                std::chrono::high_resolution_clock::now();
#endif

        // Futher work: Apply regularization modification to the normal equation, if any
        /// (4)-(7) Solve the normal equation
        if (!BA_config.use_CG)
        {
            // Option 1: Analytical solver
            model_ptr_->solveNormalEq(A11, A12, A22_blocks, b1, b2, lambda, x1, x2);
        }
        else
        {
            // Option 2: Iterative solver (conjugate gradient)
            std::pair<int,double> res = model_ptr_->solveNormalEqCG(A11, A12, A22_blocks, b1, b2, lambda, x1, x2);
            std::ofstream CG_iterations(result_dir_+"/final_results/CG_iterations.txt", std::ios::app);
            CG_iterations << "iter #" << iter << " iter_times = "
                          << res.first << " error = " << res.second << std::endl;
        }

#ifdef ENABLE_CHRONO
        std::chrono::high_resolution_clock::time_point t2_solveEqs =
                std::chrono::high_resolution_clock::now();
        auto duration_solveEqs =
                std::chrono::duration_cast<std::chrono::milliseconds>(t2_solveEqs - t1_solveEqs).count();
        count_solveEqs += 1;
        t_total_solveEqs += duration_solveEqs;
        double sec_total_solveEqs = t_total_solveEqs/1e3;
        double sec_average_solveEqs = sec_total_solveEqs/count_solveEqs;

        VLOG(1) << "[NormalEqs Solving] count_solveEqs: " << count_solveEqs;
        VLOG(1) << "[NormalEqs Solving] sec_total_solveEqs: " << std::setprecision(9) << sec_total_solveEqs;
        VLOG(1) << "[NormalEqs Solving] sec_average_solveEqs: " << std::setprecision(9) << sec_average_solveEqs;

        std::ofstream runtime_solveEqs(result_dir_+"/final_results/runtime_solveEqs.txt", std::ios::app);
        runtime_solveEqs << "iter #" << iter << " count_solveEqs = " << count_solveEqs
                         << " sec_total_solveEqs = " << std::setprecision(9) << sec_total_solveEqs
                         << " sec_average_solveEqs = " << std::setprecision(9) << sec_average_solveEqs << std::endl;
#endif

        /// (8.a) Update the parameter vector by applying the incremental vector
        // Update the intensity map with the solved perturbation
        Trajectory* traj_new_ptr = traj_ptr->clone();
        if (first_time_window_)
        {
            model_ptr_->updateTraj(traj_new_ptr, x1, 1);
        }
        else
        {
            model_ptr_->updateTraj(traj_new_ptr, x1);
        }

        // Update the gradient map with the solved perturbation
        cv::Mat Gx_new = Gx.clone();
        cv::Mat Gy_new = Gy.clone();
        model_ptr_->updateMap(Gx_new, Gy_new, x2, BA_config.damping_factor,
                              active_pix_idxes, inactive_pix_idxes);

#ifdef ENABLE_CHRONO
        static unsigned long count_obj_func = 0; // Number of times of evaluating the objective function
        static long double t_total_obj_func = 0; // [s]
        std::chrono::high_resolution_clock::time_point t1_obj_func =
                std::chrono::high_resolution_clock::now();
#endif

        /// (8.b) Evaluate the model function at the new point
        // 1. data cost
        ep_data_new = model_ptr_->evaluateDataError(traj_new_ptr, Gx_new, Gy_new, event_subset,
                                                    true, num_ev_map_new);
        // 2. regularization cost
        ep_reg_new = model_ptr_->evaluateRegError(Gx_new, Gy_new);

        // Evaluate the cost value at the new point
        if (BA_config.use_IRLS)
        {
            cost_data_new = model_ptr_->evaluateRobustDataCost(ep_data_new,
                                                               BA_config.cost_type,
                                                               BA_config.eta);
        }
        else
        {
            cost_data_new = 0.5 * ep_data_new.dot(ep_data_new);
        }
        cost_reg_new = alpha * 0.5 * ep_reg_new.dot(ep_reg_new);
        cost_new = cost_data_new + cost_reg_new;

        // Increase iteration counter
        iter += 1;

#ifdef ENABLE_CHRONO
        std::chrono::high_resolution_clock::time_point t2_obj_func =
                std::chrono::high_resolution_clock::now();
        auto duration_obj_func =
                std::chrono::duration_cast<std::chrono::milliseconds>(t2_obj_func - t1_obj_func).count();
        count_obj_func += 1;
        t_total_obj_func += duration_obj_func;
        double sec_total_obj_func = t_total_obj_func/1e3;
        double sec_average_obj_func = sec_total_obj_func/count_obj_func;

        unsigned long Np = active_pix_idxes.size();

        VLOG(1) << "[Model Function] count_obj_func: " << count_obj_func;
        VLOG(1) << "[Model Function] sec_total_obj_func: " << sec_total_obj_func;
        VLOG(1) << "[Model Function] sec_average_obj_func: " << sec_average_obj_func;
        VLOG(1) << "[Model Function] Np: " << Np;

        std::ofstream runtime_objFuncs(result_dir_+"/final_results/runtime_objFuncs.txt", std::ios::app);
        runtime_objFuncs << "iter #" << iter << " count_obj_func = " << count_obj_func
                         << " sec_total_obj_func = " << std::setprecision(9) << sec_total_obj_func
                         << " sec_average_obj_func = " << sec_average_obj_func << " Np = " << Np << std::endl;
#endif

        /// (9)-(10) If the error is less than the old error, then accept the new
        /// values of the parameters, diminish the value of lambda by a factor of
        /// 10, and start again at step 2, or else terminate
        if (cost_new < cost_min)
        {
            // Set the flag to be true
            cost_has_decreased = true;
            // Update trajectory
            delete traj_ptr;
            traj_ptr = traj_new_ptr;
            // Update map
            Gx_new.copyTo(Gx);
            Gy_new.copyTo(Gy);

            // Update lambda
            lambda = lambda/10;
            cost_min_old = cost_min;
            cost_min = cost_new;
            cost_data = cost_data_new;
            cost_reg = cost_reg_new;
            e_data_min = ep_data_new;
            e_reg_min = ep_reg_new;
            // Tolerance achieved?
            if (std::abs(1-cost_min/(cost_min_old+1e-10)) < LM_params.tol_fun)
            {
                // Relative function value changed by less than TolFun
                count_tol_fun_sat = count_tol_fun_sat + 1;
                VLOG(0) << "Relative function value changed by less than TolFun";

                // Convergence test (consecutive times TolFun has been reached)
                if (count_tol_fun_sat >= LM_params.num_times_tol_fun_sat)
                {
                    // Display results, compute output arguments if necessary
                    VLOG(0) << "Convergence at iteration # = " << iter
                            << ", c_min^2 = " << cost_min;
                    // Save the final result
                    if (record_data_)
                    {
                        saveEvoData(Gx, Gy, str_win_id, iter);
                        saveOptData(Gx, Gy, str_win_id, iter);
                    }
                    // Return the optimized error vector
                    return e_data_min;
                }
            }
        }
        else
        {
            // If the new error is greater than the old error, then revert to
            // the old parameter values, increase the value of lambda by a
            // factor of 10, and try again from the modified normal equations
            // with the new lambda.
            cost_has_decreased = false;
            lambda *= 10;
            // Reset convergence counter
            count_tol_fun_sat = 0;
        }
    }
    // Forced termination
    // Reached this point if reached maximum number of iterations or maximum
    // value of exploration parameter or cost function smaller than lower bound
    VLOG(0) << "Forced termination in LM, c_min^2 = " << cost_min;

    // Save the final result
    if (record_data_)
    {
        saveEvoData(Gx, Gy, str_win_id, iter);
        saveOptData(Gx, Gy, str_win_id, iter);
    }

    // Return the optimized error vector
    return e_data_min;
}

void EMBA::saveEvoData(const cv::Mat& Gx, const cv::Mat& Gy,
                       const std::string str_win_id, const size_t iter)
{
    // Write the evolving maps into the disk (for replay in the future)
    // Gradient maps
    cv::Mat Gx_disp, Gy_disp;
    image_util::normalizeRobust(Gx, Gx_disp, 0.1); // Robust normalization
    image_util::normalizeRobust(Gy, Gy_disp, 0.1); // Robust normalization
    std::stringstream ss_Gx, ss_Gy;
    ss_Gx << result_dir_ + "/Gx_evo/" << str_win_id << "Gx_evo_"
          << std::setfill('0') << std::setw(4) << iter << ".png";
    ss_Gy << result_dir_ + "/Gy_evo/" << str_win_id << "Gy_evo_"
          << std::setfill('0') << std::setw(4) << iter << ".png";
    cv::imwrite(ss_Gx.str(), Gx_disp);
    cv::imwrite(ss_Gy.str(), Gy_disp);

    // Merged Gx and Gy
    cv::Mat magnitude, oritentation;
    cv::cartToPolar(Gx, Gy, magnitude, oritentation, true); // The range of angle is [0, 360)
    oritentation = 0.5 * oritentation; // OpenCV HSV, the range of H is [0, 179]
    cv::Mat H, S, V; 

    cv::normalize(oritentation, H, 0, 179, cv::NORM_MINMAX, CV_8UC1);
    S = cv::Mat::zeros(Gx.rows, Gx.cols, CV_8UC1);
    S.setTo(255);

    // Convert the Value channel into the exp scale
    cv::normalize(magnitude, V, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    std::vector<cv::Mat> HSV_channels;
    HSV_channels.push_back(H);
    HSV_channels.push_back(S);
    HSV_channels.push_back(V);
    cv::Mat G_hsv, G_hsv_disp;
    cv::merge(HSV_channels, G_hsv);
    std::stringstream ss_G_hsv;
    ss_G_hsv << result_dir_ + "/G_hsv_evo/" << str_win_id << "G_hsv_evo_"
             << std::setfill('0') << std::setw(4) << iter << ".png";
    cv::cvtColor(G_hsv, G_hsv_disp, cv::COLOR_HSV2BGR);
    cv::imwrite(ss_G_hsv.str(), G_hsv_disp);

    // Reconstructed intensity map
    // Reconstruct the intensity map using the evolving gradient maps
    std::vector<cv::Mat> gradient_channels;
    gradient_channels.push_back(Gx);
    gradient_channels.push_back(Gy);
    cv::Mat gradient_map;
    cv::merge(gradient_channels, gradient_map);
    cv::Mat map_poisson = poisson_reconstruction::reconstructFromGradient(gradient_map);
    // Save into the disk
    cv::Mat map_poisson_disp;
    image_util::normalizeRobust(map_poisson, map_poisson_disp, 0.1); // Robust normalization
    std::stringstream ss_map_poisson;
    ss_map_poisson << result_dir_ << "/map_poisson_evo/" << str_win_id << "map_poisson_evo_"
                   << std::setfill('0') << std::setw(4) << iter << ".png";
    cv::imwrite(ss_map_poisson.str(), map_poisson_disp);
}

void EMBA::saveOptData(const cv::Mat& Gx, const cv::Mat& Gy,
                       const std::string str_win_id, const size_t iter)
{
    // Write the evolving maps into the disk (for replay in the future)
    // Gradient maps
    cv::Mat Gx_disp, Gy_disp;
    image_util::normalizeRobust(Gx, Gx_disp, 0.1); // Robust normalization
    image_util::normalizeRobust(Gy, Gy_disp, 0.1); // Robust normalization
    std::stringstream ss_Gx, ss_Gy;
    ss_Gx << result_dir_ + "/map_opt/" << str_win_id << "Gx_opt_"
          << std::setfill('0') << std::setw(4) << iter << ".png";
    ss_Gy << result_dir_ + "/map_opt/" << str_win_id << "Gy_opt_"
          << std::setfill('0') << std::setw(4) << iter << ".png";
    cv::imwrite(ss_Gx.str(), Gx_disp);
    cv::imwrite(ss_Gy.str(), Gy_disp);

    // Merged Gx and Gy
    cv::Mat magnitude, oritentation;
    cv::cartToPolar(Gx, Gy, magnitude, oritentation, true); // The range of angle is [0, 360)
    oritentation = 0.5 * oritentation; // OpenCV HSV, the range of H is [0, 179]
    cv::Mat H, S, V;
    cv::normalize(oritentation, H, 0, 179, cv::NORM_MINMAX, CV_8UC1);
    S = cv::Mat::zeros(Gx.rows, Gx.cols, CV_8UC1);
    S.setTo(255);
    cv::normalize(magnitude, V, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    std::vector<cv::Mat> HSV_channels;
    HSV_channels.push_back(H);
    HSV_channels.push_back(S);
    HSV_channels.push_back(V);
    cv::Mat G_hsv, G_hsv_disp;
    cv::merge(HSV_channels, G_hsv);
    std::stringstream ss_G_hsv;
    ss_G_hsv << result_dir_ + "/map_opt/" << str_win_id << "G_hsv_opt_"
             << std::setfill('0') << std::setw(4) << iter << ".png";
    cv::cvtColor(G_hsv, G_hsv_disp, cv::COLOR_HSV2BGR);
    cv::imwrite(ss_G_hsv.str(), G_hsv_disp);

    // Reconstructed intensity map
    // Reconstruct the intensity map using the evolving gradient maps
    std::vector<cv::Mat> gradient_channels;
    gradient_channels.push_back(Gx);
    gradient_channels.push_back(Gy);
    cv::Mat gradient_map;
    cv::merge(gradient_channels, gradient_map);
    cv::Mat map_poisson = poisson_reconstruction::reconstructFromGradient(gradient_map);
    // Save into the disk
    cv::Mat map_poisson_disp;
    image_util::normalizeRobust(map_poisson, map_poisson_disp, 0.1); // Robust normalization
    std::stringstream ss_map_poisson;
    ss_map_poisson << result_dir_ << "/map_opt/" << str_win_id << "map_poisson_opt_"
                   << std::setfill('0') << std::setw(4) << iter << ".png";
    cv::imwrite(ss_map_poisson.str(), map_poisson_disp);
}

}

