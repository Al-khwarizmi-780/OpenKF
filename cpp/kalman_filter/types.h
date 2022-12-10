///
/// Copyright 2022 CodingCorner
///
/// Use of this source code is governed by an GPL-3.0 - style
/// license that can be found in the LICENSE file or at
/// https ://https://opensource.org/licenses/GPL-3.0.
///
/// @author Mohanad Youssef <mohanad.magdy.hammad@gmail.com>
/// @file types.h
///

#ifndef __KALMAN_FILTER_TYPES_H__
#define __KALMAN_FILTER_TYPES_H__

#include <stdint.h>
#include <Eigen/Dense>

using float32_t = float;

template<size_t ROW, size_t COL>
using Matrix = Eigen::Matrix<float32_t, ROW, COL>;

template<size_t ROW>
using Vector = Eigen::Matrix<float32_t, ROW, 1>;

#endif // __KALMAN_FILTER_TYPES_H__
