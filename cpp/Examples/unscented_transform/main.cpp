///
/// Copyright 2022 CodingCorner
///
/// Use of this source code is governed by an MIT - style
/// license that can be found in the LICENSE file or at
/// https ://opensource.org/licenses/MIT.
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

Vector<DIM_1> function1(const Vector<DIM_1> & x)
{
    Vector<DIM_1> y;
    y[0] = x[0] * x[0];
    return y;
}

Vector<DIM_2> function2(const Vector<DIM_2> & x)
{
    Vector<DIM_2> y;
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

    Vector<DIM_1> x;
    x << 0.0F;

    Matrix<DIM_1, DIM_1> P;
    P << 0.5F;

    kf::UnscentedTransform<DIM_1> UT(function1, x, P, 0.0F);

    UT.showSummary();
    std::cout << " End of Example 1: ===========================" << std::endl;
}

void runExample2()
{
    std::cout << " Start of Example 2: ===========================" << std::endl;

    Vector<DIM_2> x;
    x << 2.0F, 1.0F;

    Matrix<DIM_2, DIM_2> P;
    P << 0.1F, 0.0F, 0.0F, 0.1F;

    kf::UnscentedTransform<DIM_2> UT(function2, x, P, 0.0F);

    UT.showSummary();
    std::cout << " End of Example 2: ===========================" << std::endl;
}
