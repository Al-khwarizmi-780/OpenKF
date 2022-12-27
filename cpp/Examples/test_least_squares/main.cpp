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
#include "kalman_filter/util.h"

void runExample1();
void runExample2();
void runExample3();
void runExample4();

int main(int argc, char ** argv)
{
    runExample1();
    runExample2();
    runExample3();
    runExample4();

    return 0;
}

void runExample1()
{
    std::cout << " Start of Example 1: ===========================" << std::endl;

    kf::Matrix<3, 3> A;
    A << 3.0, 2.0, 1.0,
         2.0, 3.0, 4.0,
         1.0, 4.0, 3.0;

    kf::Matrix<2, 3> B;
    B << 5.0, 6.0, 7.0,
         8.0, 9.0, 10.0;

    kf::util::JointRows<3, 2, 3> jmat(A, B);
    auto AB = jmat.jointMatrix();

    std::cout << "Joint Rows: AB = \n" << AB << std::endl;

    std::cout << " End of Example 1: ===========================" << std::endl;
}

void runExample2()
{
    std::cout << " Start of Example 2: ===========================" << std::endl;

    kf::Matrix<3, 3> A;
    A << 3.0, 2.0, 1.0,
        2.0, 3.0, 4.0,
        1.0, 4.0, 3.0;

    kf::Matrix<3, 2> B;
    B << 5.0, 6.0,
         7.0, 8.0,
         9.0, 10.0;

    kf::util::JointCols<3, 3, 2> jmat(A, B);
    auto AB = jmat.jointMatrix();

    std::cout << "Joint Columns: AB = \n" << AB << std::endl;

    std::cout << " End of Example 2: ===========================" << std::endl;
}

void runExample3()
{
    std::cout << " Start of Example 2: ===========================" << std::endl;

    kf::Matrix<3, 3> A;
    A << 1.0, -2.0, 1.0,
        0.0, 1.0, 6.0,
        0.0, 0.0, 1.0;

    kf::Matrix<3, 1> b;
    b << 4.0, -1.0, 2.0;

    auto x = kf::util::backwardSubstitute<3, 1>(A, b);

    std::cout << "Backward Substitution: x = \n" << x << std::endl;

    std::cout << " End of Example 2: ===========================" << std::endl;
}

void runExample4()
{
    std::cout << " Start of Example 2: ===========================" << std::endl;

    kf::Matrix<3, 3> A;
    A << 1.0, 0.0, 0.0,
         -2.0, 1.0, 0.0,
         1.0, 6.0, 1.0;

    kf::Matrix<3, 1> b;
    b << 4.0, -1.0, 2.0;

    auto x = kf::util::forwardSubstitute<3, 1>(A, b);

    std::cout << "Forward Substitution: x = \n" << x << std::endl;

    std::cout << " End of Example 2: ===========================" << std::endl;
}
