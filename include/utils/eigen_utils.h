#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>

#include <unsupported/Eigen/SparseExtra>

namespace eigen_utils {

// Concatenate Eigen sparse matrix, get from:
// https://github.com/libigl/libigl/blob/main/include/igl/cat.h

/// Perform concatenation of a two _sparse_ matrices along a single dimension
/// If dim == 1, then C = [A;B]; If dim == 2 then C = [A B].
/// This is an attempt to act like matlab's cat function.
///
/// @tparam  double  double data type for sparse matrices like double or int
/// @tparam  Mat  matrix type for all matrices (e.g. MatrixXd, SparseMatrix)
/// @tparam  MatC  matrix type for output matrix (e.g. MatrixXd) needs to support
///     resize
/// @param[in]  dim  dimension along which to concatenate, 1 or 2
/// @param[in]  A  first input matrix
/// @param[in]  B  second input matrix
/// @param[out]  C  output matrix
///
void catSpMat(const int dim,
              const Eigen::SparseMatrix<double> & A,
              const Eigen::SparseMatrix<double> & B,
              Eigen::SparseMatrix<double> & C);

// Given the vector of diagnol elements, return a (dense) diagonal matrix
Eigen::MatrixXd diagMat(const Eigen::VectorXd diag);

// Tailored for 2D matrix
Eigen::Matrix2d diagMat2d(const Eigen::Vector2d diag);

// Given the vector of diagnol elements, return a (sparse) diagonal matrix
Eigen::SparseMatrix<double> diagSpMat(const Eigen::VectorXd diag);
}
