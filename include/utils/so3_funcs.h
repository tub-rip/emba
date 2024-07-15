#pragma once

#include "utils/trajectory.h"

namespace so3_utils {

// There is no cotangent implementation in C++,
// we need to approximate it by cos(x)/sin(x).
// https://stackoverflow.com/questions/3738384/stable-cotangent
inline double cot(const double x)
{
    const double res = cos(x)/sin(x);
    return res;
}

// Given a rotation vector, return the corresponding skew symmetric matrix
// (hat operator)
inline Eigen::Matrix3d v2skew(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d m;
    m <<  0.0, -v(2),  v(1),
         v(2),   0.0, -v(0),
        -v(1),  v(0),  0.0;
    return m;
}

// Given a skew symmetric matrix, return the corresponding rotation vector
// (vee operator)
inline Eigen::Vector3d skew2v(const Eigen::Matrix3d& m)
{
    Eigen::Vector3d v;
    v(0) = 0.5*(m(2,1)-m(1,2));
    v(1) = 0.5*(m(0,2)-m(2,0));
    v(2) = 0.5*(m(1,0)-m(0,1));
    return v;
}

// Given a rotation vector, return the corresponding left Jacobian
Eigen::Matrix3d Jl(const Eigen::Vector3d& rotv)
{
    const double phi = rotv.norm(); // Rotation angle
    const Eigen::Vector3d a = rotv/phi; // Rotation axis (unit vector)
    const double temp = sin(phi)/phi;
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity(); // 3X3 identity matrix
    const Eigen::Matrix3d Jl = temp*I + (1-temp)*(a*a.transpose()) + (1-cos(phi))/phi*v2skew(a);
    return Jl;
}

// Given a rotation vector, return corresponding inverse left Jacobian
Eigen::Matrix3d Jl_inv(const Eigen::Vector3d& rotv)
{
    const double phi = rotv.norm(); // Rotation angle
    const Eigen::Vector3d a = rotv/phi; // Rotation axis (unit vector)
    const double temp1 = phi/2;
    const double temp2 = temp1*cot(temp1);
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity(); // 3X3 identity matrix
    const Eigen::Matrix3d Jl_inv = temp2*I + (1-temp2)*(a*a.transpose()) - temp1*v2skew(a);
    return Jl_inv;
}

}
