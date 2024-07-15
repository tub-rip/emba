#include "utils/so3_funcs.h"
#include "utils/trajectory.h"

#include <iostream>

int main(int argc, char *argv[])
{
    Eigen::Vector3d v(0.5377,1.8339,-2.2588);
    std::cout << "v = \n" << v.transpose() << std::endl;
    Eigen::Matrix3d m = so3_utils::v2skew(v);
    std::cout << "v2skew(v) = \n" << m << std::endl;
    v = so3_utils::skew2v(m);
    std::cout << "skew2v(m) = \n" << v.transpose() << std::endl;
    Eigen::Matrix3d Jl_inv = so3_utils::Jl_inv(v);
    std::cout << "Jl_inv = \n" << Jl_inv << std::endl;
    Eigen::Matrix3d Jl = so3_utils::Jl(v);
    std::cout << "Jl = \n" << Jl << std::endl;
    return 0;
}
