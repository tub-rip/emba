#pragma once

#include <set>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "emba/event_map.h"
#include "utils/trajectory.h"
#include "utils/event_pano_warper.h"
#include "image_rec/poisson_reconstruction.h"

namespace EMBA {

typedef std::vector<dvs_msgs::Event> EventPacket;

typedef Eigen::VectorXd VecXd;
typedef Eigen::MatrixXd MatXd;
typedef Eigen::Matrix2d Mat2d;
typedef Eigen::Matrix<double,6,6> Mat6d;
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic> DiagMat;
typedef Eigen::Triplet<double> Triplet; // used to initialize the sparse matrix

// Abstract class for the model function
class Model
{
public:
    // Virtual deconstructor
    virtual ~Model () {}

    // Update the camera trajectory with the solved perturbations
    // (the same for OEGM and LEGM, so not virtual)
    void updateTraj(Trajectory* traj_ptr, const VecXd& x1);

    // Update the camera trajectory with the solved perturbations,
    // but for the first time window (fix the first control pose)
    // (the same for OEGM and LEGM, so not virtual)
    void updateTraj(Trajectory* traj_ptr, const VecXd& x1, const size_t idx_cp_beg);

    // Convert 2D pixel locations (on the pano map) to 1D index (row-major storage)
    inline size_t pxToIdx(int x, int y) const
    { return x + y * event_warper_ptr_->panoMapWidth(); }

    // Draw Camera FOV
    void drawSensorFOV(cv::Mat& canvas,
                       const Sophus::SO3d& R,
                       const cv::MarkerTypes& marker_type,
                       const int marker_size,
                       const cv::Vec3i& color)
    {
        event_warper_ptr_->drawSensorFOV(canvas, R, marker_type, marker_size, color);
    }

    // Convert 1D index (row-major storage) to 2D pixel locations (on the pano map)
    inline cv::Point2i idxToPx(const size_t idx) const
    {
        const int y = idx/event_warper_ptr_->panoMapWidth();
        const int x = idx - y * event_warper_ptr_->panoMapWidth();
        return cv::Point2i(x,y);
    }

protected:
    // Event camera contrast threshold
    double C_th_;

    // Event warper: warp events onto the pano map, and compute derivatives
    dvs::EventWarper* event_warper_ptr_;
};

// Model function with the linearized event generation model (LEGM)
class LEGM: public Model
{
public:
    // Constructor
    LEGM(const sensor_msgs::CameraInfo& camera_info_msg, double C_th,
         int pano_width, int pano_height);

    // Deconstructor
    ~LEGM () { delete event_warper_ptr_; }

    // Evaluate the error terms caused by the event data
    VecXd evaluateDataError(Trajectory* traj_ptr, const cv::Mat& Gx, const cv::Mat& Gy,
                            const EventPacket &events, bool eval_deriv, cv::Mat& num_ev_map);

    // Evaluate the error terms caused by the L2 regularization on the gradient map
    VecXd evaluateRegError(const cv::Mat& Gx, const cv::Mat& Gy);

    // if IRLS is enabled, evaluate the robust cost value
    double evaluateRobustDataCost(const VecXd& ep, const std::string cost_type, const double a);

    // Form the normal equation using the error and derivatives from the data part
    void formNormalEq(MatXd& A11, MatXd& A12, std::vector<Mat2d>& A22_blocks,
                      VecXd& b1, VecXd& b2, const VecXd& ep, const int num_ctrl_poses,
                      const cv::Mat& num_ev_map, const int thres_valid_pixel,
                      std::set<size_t>& active_pix_idxes, std::set<size_t>& inactive_pix_idxes);

    // Form normal equation with robust cost function
    void formNormalEqIRLS(MatXd& A11, MatXd& A12, std::vector<Mat2d>& A22_blocks,
                          VecXd& b1, VecXd& b2, const VecXd& ep, const int num_ctrl_poses,
                          const cv::Mat& num_ev_map, const int thres_valid_pixel,
                          std::set<size_t>& active_pix_idxes, std::set<size_t>& inactive_pix_idxes,
                          const std::string cost_type, const double a);

    // Apply the effect of the L2 regularization to the formed normal equation
    void applyL2Reg(std::vector<Mat2d>& A22_blocks, VecXd& b2,
                    const std::set<size_t> &active_pix_idxes,
                    const double alpha, const cv::Mat& Gx, const cv::Mat& Gy);

    // Solve for the optimal perturbations on the camera poses and valid pixels
    void solveNormalEq(const MatXd& A11, const MatXd& A12, const std::vector<Mat2d>& A22_blocks,
                       const VecXd& b1, const VecXd& b2, const double lambda,
                       VecXd& x1, VecXd& x2);

    // Solve the normal euqation using the conjugate gradient method
    std::pair<int, double> solveNormalEqCG(const MatXd& A11, const MatXd& A12,
                                           const std::vector<Mat2d>& A22_blocks,
                                           const VecXd& b1, const VecXd& b2,
                                           const double lambda, VecXd& x1, VecXd& x2);

    // Update the map (both active and inactive pixels)
    void updateMap(cv::Mat& Gx_new, cv::Mat& Gy_new,
                   const VecXd& x2, const double damping_factor,
                   const std::set<size_t>& active_pix_idxes,
                   const std::set<size_t>& inactive_pix_idxes);

    // Recover the sparse A22 from its nonzero blocks
    SpMat recoverA22FromBlocks(const std::vector<Mat2d>& A22_blocks);

private:
    // The event time map for LEGM, used to form the normal equation
    EventMap<State_LEGM> event_map_;
};

}
