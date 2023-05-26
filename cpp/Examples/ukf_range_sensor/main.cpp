///
/// Copyright 2022 Mohanad Youssef (Al-khwarizmi)
///
/// Use of this source code is governed by an GPL-3.0 - style
/// license that can be found in the LICENSE file or at
/// https://opensource.org/licenses/GPL-3.0
///
/// @author Mohanad Youssef <mohanad.magdy.hammad@gmail.com>
/// @file main.cpp
///

#include <iostream>
#include <stdint.h>

#include "kalman_filter/types.h"
#include "kalman_filter/unscented_kalman_filter.h"

static constexpr size_t DIM_X{ 4 };
static constexpr size_t DIM_V{ 4 };
static constexpr size_t DIM_Z{ 2 };
static constexpr size_t DIM_N{ 2 };

void runExample1();

kf::Vector<DIM_X> funcF(const kf::Vector<DIM_X> & x, const kf::Vector<DIM_V> & v)
{
    kf::Vector<DIM_X> y;
    y[0] = x[0] + x[2] + v[0];
    y[1] = x[1] + x[3] + v[1];
    y[2] = x[2] + v[2];
    y[3] = x[3] + v[3];
    return y;
}

kf::Vector<DIM_Z> funcH(const kf::Vector<DIM_X> & x, const kf::Vector<DIM_N> & n)
{
    kf::Vector<DIM_Z> y;

    kf::float32_t px{ x[0] + n[0] };
    kf::float32_t py{ x[1] + n[1] };

    y[0] = std::sqrt((px * px) + (py * py));
    y[1] = std::atan(py / (px + std::numeric_limits<kf::float32_t>::epsilon()));
    return y;
}

int main(int argc, char ** argv)
{
    // example 1
    runExample1();

    return 0;
}

void runExample1()
{
    std::cout << " Start of Example 1: ===========================" << std::endl;

    kf::Vector<DIM_X> x;
    x << 2.0F, 1.0F, 0.0F, 0.0F;

    kf::Matrix<DIM_X, DIM_X> P;
    P << 0.01F, 0.0F, 0.0F, 0.0F,
         0.0F, 0.01F, 0.0F, 0.0F,
         0.0F, 0.0F, 0.05F, 0.0F,
         0.0F, 0.0F, 0.0F, 0.05F;

    kf::Matrix<DIM_V, DIM_V> Q;
    Q << 0.05F, 0.0F, 0.0F, 0.0F,
        0.0F, 0.05F, 0.0F, 0.0F,
        0.0F, 0.0F, 0.1F, 0.0F,
        0.0F, 0.0F, 0.0F, 0.1F;

    kf::Matrix<DIM_N, DIM_N> R;
    R << 0.01F, 0.0F, 0.0F, 0.01F;

    kf::Vector<DIM_Z> z;
    z << 2.5, 0.05;

    kf::UnscentedKalmanFilter<DIM_X, DIM_Z, DIM_V, DIM_N> ukf;

    ukf.vecX() = x;
    ukf.matP() = P;

    ukf.setCovarianceQ(Q);
    ukf.setCovarianceR(R);

    ukf.predictUKF(funcF);

    std::cout << "x = \n" << ukf.vecX() << std::endl;
    std::cout << "P = \n" << ukf.matP() << std::endl;

    // Expectation from the python results:
    // =====================================
    // x =
    //     [2.0 1.0 0.0 0.0]
    // P =
    //     [[0.11  0.00  0.05  0.00]
    //      [0.00  0.11  0.00  0.05]
    //      [0.05  0.00  0.15  0.00]
    //      [0.00  0.05  0.00  0.15]]

    ukf.correctUKF(funcH, z);

    std::cout << "x = \n" << ukf.vecX() << std::endl;
    std::cout << "P = \n" << ukf.matP() << std::endl;

    // Expectations from the python results:
    // ======================================
    // x =
    //     [ 2.554  0.356  0.252 -0.293]
    // P =
    //     [[ 0.01  -0.001  0.005 -0.    ]
    //      [-0.001  0.01 - 0.     0.005 ]
    //      [ 0.005 - 0.     0.129 - 0.  ]
    //      [-0.     0.005 - 0.     0.129]]

    std::cout << " End of Example 1: ===========================" << std::endl;
}

