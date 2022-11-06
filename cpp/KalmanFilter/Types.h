///
/// @author Mohanad Youssef
/// @file KalmanFilterExercise/KalmanFilter/Types.h
///

#ifndef __KALMAN_FILTER_TYPES_H__
#define __KALMAN_FILTER_TYPES_H__

#include <stdint.h>
#include <Eigen/Dense>

namespace kf
{
    using float32_t = float;

    template<size_t ROW, size_t COL>
    using Matrix = Eigen::Matrix<float32_t, ROW, COL>;

    template<size_t ROW>
    using Vector = Eigen::Matrix<float32_t, ROW, 1>;

}

#endif // __KALMAN_FILTER_TYPES_H__
