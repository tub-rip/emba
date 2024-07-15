#include "utils/eigen_utils.h"

void eigen_utils::catSpMat(const int dim,
                           const Eigen::SparseMatrix<double> & A,
                           const Eigen::SparseMatrix<double> & B,
                           Eigen::SparseMatrix<double> & C)
{
    assert(dim == 1 || dim == 2);
    using namespace Eigen;
    // Special case if B or A is empty
    if(A.size() == 0)
    {
        C = B;
        return;
    }
    if(B.size() == 0)
    {
        C = A;
        return;
    }

    // This is faster than using DynamicSparseMatrix or setFromTriplets
    C = SparseMatrix<double>(
                dim == 1 ? A.rows()+B.rows() : A.rows(),
                dim == 1 ? A.cols()          : A.cols()+B.cols());
    Eigen::VectorXi per_col = Eigen::VectorXi::Zero(C.cols());
    if(dim == 1)
    {
        assert(A.outerSize() == B.outerSize());
        for(int k = 0;k<A.outerSize();++k)
        {
            for(typename SparseMatrix<double>::InnerIterator it (A,k); it; ++it)
            {
                per_col(k)++;
            }
            for(typename SparseMatrix<double>::InnerIterator it (B,k); it; ++it)
            {
                per_col(k)++;
            }
        }
    }else
    {
        for(int k = 0;k<A.outerSize();++k)
        {
            for(typename SparseMatrix<double>::InnerIterator it (A,k); it; ++it)
            {
                per_col(k)++;
            }
        }
        for(int k = 0;k<B.outerSize();++k)
        {
            for(typename SparseMatrix<double>::InnerIterator it (B,k); it; ++it)
            {
                per_col(A.cols() + k)++;
            }
        }
    }
    C.reserve(per_col);
    if(dim == 1)
    {
        for(int k = 0;k<A.outerSize();++k)
        {
            for(typename SparseMatrix<double>::InnerIterator it (A,k); it; ++it)
            {
                C.insert(it.row(),k) = it.value();
            }
            for(typename SparseMatrix<double>::InnerIterator it (B,k); it; ++it)
            {
                C.insert(A.rows()+it.row(),k) = it.value();
            }
        }
    }else
    {
        for(int k = 0;k<A.outerSize();++k)
        {
            for(typename SparseMatrix<double>::InnerIterator it (A,k); it; ++it)
            {
                C.insert(it.row(),k) = it.value();
            }
        }
        for(int k = 0;k<B.outerSize();++k)
        {
            for(typename SparseMatrix<double>::InnerIterator it (B,k); it; ++it)
            {
                C.insert(it.row(),A.cols()+k) = it.value();
            }
        }
    }
    C.makeCompressed();
}

Eigen::MatrixXd eigen_utils::diagMat(const Eigen::VectorXd diag)
{
    const size_t dim = diag.size();
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(dim,dim);
    for (size_t i = 0; i < dim; i++)
    {
        D(i,i) = diag(i);
    }
    return D;
}

Eigen::Matrix2d eigen_utils::diagMat2d(const Eigen::Vector2d diag)
{
    Eigen::Matrix2d D = Eigen::Matrix2d::Zero();
    for (size_t i = 0; i < 2; i++)
    {
        D(i,i) = diag(i);
    }
    return D;
}

Eigen::SparseMatrix<double> eigen_utils::diagSpMat(const Eigen::VectorXd diag)
{
    const size_t dim = diag.size();
    std::vector<Eigen::Triplet<double>> nonzero_entries;
    nonzero_entries.reserve(dim);

    // Initialize sparse matrix from a list of non-zero elements
    Eigen::SparseMatrix<double> D(dim, dim);
    for (size_t i = 0; i < dim; i++)
    {
        nonzero_entries.push_back(Eigen::Triplet<double>(i,i,diag(i)));
    }
    D.setFromTriplets(nonzero_entries.begin(), nonzero_entries.end());
    D.makeCompressed();
    return D;

}
