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

#include "types.h"
#include "kalman_filter/square_root_ukf.h"

static constexpr size_t DIM_X{ 2 };
static constexpr size_t DIM_Z{ 2 };

void runExample1();

kf::Vector<DIM_X> funcF(const kf::Vector<DIM_X> & x)
{
    return x;
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

    // initializations
    //x0 = np.array([1.0, 2.0])
    //P0 = np.array([[1.0, 0.5], [0.5, 1.0]])
    //Q = np.array([[0.5, 0.0], [0.0, 0.5]])

    //z = np.array([1.2, 1.8])
    //R = np.array([[0.3, 0.0], [0.0, 0.3]])

    kf::Vector<DIM_X> x;
    x << 1.0F, 2.0F;

    kf::Matrix<DIM_X, DIM_X> P;
    P << 1.0F, 0.5F,
         0.5F, 1.0F;

    kf::Matrix<DIM_X, DIM_X> Q;
    Q << 0.5F, 0.0F,
         0.0F, 0.5F;

    kf::Vector<DIM_Z> z;
    z << 1.2F, 1.8F;

    kf::Matrix<DIM_Z, DIM_Z> R;
    R << 0.3F, 0.0F,
         0.0F, 0.3F;

    kf::SquareRootUKF<DIM_X, DIM_Z> srUkf;
    srUkf.initialize(x, P, Q, R);

    srUkf.predictSRUKF(funcF);

    std::cout << "x = \n" << srUkf.vecX() << std::endl;
    std::cout << "P = \n" << srUkf.matP() << std::endl;

    // Expectation from the python results:
    // =====================================
    //x1 =
    //    [1. 2.]
    //P1 =
    //    [[1.5 0.5]
    //     [0.5 1.5]]

    srUkf.correctSRUKF(funcF, z);

    std::cout << "x = \n" << srUkf.vecX() << std::endl;
    std::cout << "P = \n" << srUkf.matP() << std::endl;

    // Expectations from the python results:
    // ======================================
    // x =
    //     [1.15385 1.84615]
    // P =
    //     [[ 0.24582 0.01505 ]
    //      [ 0.01505 0.24582 ]]

    std::cout << " End of Example 1: ===========================" << std::endl;
}

