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

static constexpr size_t DIM_X{ 2 };
static constexpr size_t DIM_Z{ 1 };
static constexpr float32_t T{ 1.0F };
static constexpr float32_t Q11{ 0.1F }, Q22{ 0.1F };

static kf::KalmanFilter<DIM_X, DIM_Z> kalmanfilter;

void executePredictionStep();
void executeCorrectionStep();

int main(int argc, char ** argv)
{
    executePredictionStep();
    executeCorrectionStep();

    return 0;
}

void executePredictionStep()
{
    kalmanfilter.vecX() << 0.0F, 2.0F;
    kalmanfilter.matP() << 0.1F, 0.0F, 0.0F, 0.1F;

    Matrix<DIM_X, DIM_X> F; // state transition matrix
    F << 1.0F, T, 0.0F, 1.0F;

    Matrix<DIM_X, DIM_X> Q; // process noise covariance
    Q(0, 0) = (Q11 * T) + (Q22 * (std::pow(T, 3) / 3.0F));
    Q(0, 1) = Q(1, 0) = Q22 * (std::pow(T, 2) / 2.0F);
    Q(1, 1) = Q22 * T;

    kalmanfilter.predict(F, Q); // execute prediction step

    std::cout << "\npredicted state vector = \n" << kalmanfilter.vecX() << "\n";
    std::cout << "\npredicted state covariance = \n" << kalmanfilter.matP() << "\n";
}

void executeCorrectionStep()
{
    Vector<DIM_Z> vecZ;
    vecZ << 2.25F;

    Matrix<DIM_Z, DIM_Z> matR;
    matR << 0.01F;

    Matrix<DIM_Z, DIM_X> matH;
    matH << 1.0F, 0.0F;

    kalmanfilter.correct(vecZ, matR, matH);

    std::cout << "\ncorrected state vector = \n" << kalmanfilter.vecX() << "\n";
    std::cout << "\ncorrected state covariance = \n" << kalmanfilter.matP() << "\n";
}
