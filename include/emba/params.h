#pragma once
#include <string>

struct LMSettings
{
    // Maximum number of iterations allowed
    int max_num_iter;
    // Tolerance in the cost function to claim convergence
    double tol_fun;
    // % Number of consecutive times that tolerance must be satisfied to claim convergence
    int num_times_tol_fun_sat;
};

struct BASettings
{
    // Contrast threshold
    double C_th;

    // If the initial map is not available? Start from sctrach?
    bool init_map_available;

    // Use the conjugate gradient (CG) solver?
    bool use_CG;

    // Use robust cost function (IRLS)?
    bool use_IRLS;

    // Cost funtion type: quadratic || huber || cauchy
    std::string cost_type;

    // The parameter of the Huber and Cauchy cost functions
    double eta;

    // Share a common pose for a event batch
    int event_batch_size;

    // Event sample rate
    int event_sampling_rate;

    // Map size
    int pano_height, pano_width;

    // Event number threshold for valid pixel selection
    int thres_valid_pixel;

    // Damping factor on the map updating
    double damping_factor;

    // Weight of the L2 regularizer
    double alpha;


    // Size of the time window (sec)
    double time_window_size;

    // Stride of sliding window (sec)
    double sliding_window_stride;

    // Time gap between two control points (knots)
    double dt_knots;
};
