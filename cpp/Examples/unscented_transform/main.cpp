///
/// Copyright 2022 CodingCorner
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
#include "kalman_filter/kalman_filter.h"

#include "kalman_filter/unscented_transform.h"

static constexpr size_t DIM_1{ 1 };
static constexpr size_t DIM_2{ 2 };

void runExample1();
void runExample2();

kf::Vector<DIM_1> function1(const kf::Vector<DIM_1> & x)
{
    kf::Vector<DIM_1> y;
    y[0] = x[0] * x[0];
    return y;
}

kf::Vector<DIM_2> function2(const kf::Vector<DIM_2> & x)
{
    kf::Vector<DIM_2> y;
    y[0] = x[0] * x[0];
    y[1] = x[1] * x[1];
    return y;
}

int main(int argc, char ** argv)
{
    // example 1
    runExample1();

    // example 2
    runExample2();

    return 0;
}

void runExample1()
{
    std::cout << " Start of Example 1: ===========================" << std::endl;

    kf::Vector<DIM_1> x;
    x << 0.0F;

    kf::Matrix<DIM_1, DIM_1> P;
    P << 0.5F;

    kf::UnscentedTransform<DIM_1> UT;
    UT.compute(x, P, 0.0F);

    kf::Vector<DIM_1> vecY;
    kf::Matrix<DIM_1, DIM_1> matPyy;

    UT.transform(function1, vecY, matPyy);

    UT.showSummary();
    std::cout << "vecY: \n" << vecY << "\n";
    std::cout << "matPyy: \n" << matPyy << "\n";

    std::cout << " End of Example 1: ===========================" << std::endl;
}

void runExample2()
{
    std::cout << " Start of Example 2: ===========================" << std::endl;

    kf::Vector<DIM_2> x;
    x << 2.0F, 1.0F;

    kf::Matrix<DIM_2, DIM_2> P;
    P << 0.1F, 0.0F, 0.0F, 0.1F;

    kf::UnscentedTransform<DIM_2> UT;
    UT.compute(x, P, 0.0F);

    kf::Vector<DIM_2> vecY;
    kf::Matrix<DIM_2, DIM_2> matPyy;

    UT.transform(function2, vecY, matPyy);

    UT.showSummary();
    std::cout << "vecY: \n" << vecY << "\n";
    std::cout << "matPyy: \n" << matPyy << "\n";

    std::cout << " End of Example 2: ===========================" << std::endl;
}
