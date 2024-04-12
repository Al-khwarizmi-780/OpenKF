///
/// Copyright 2022 Mohanad Youssef (Al-khwarizmi)
///
/// Use of this source code is governed by an GPL-3.0 - style
/// license that can be found in the LICENSE file or at
/// https://opensource.org/licenses/GPL-3.0
///
/// @author Mohanad Youssef <mohanad.magdy.hammad@gmail.com>
/// @file types.h
///

#ifndef OPENKF_TYPES_H
#define OPENKF_TYPES_H

#include <stdint.h>
#include <Eigen/Dense>

namespace kf
{
	using float32_t = float;

	template<int32_t ROW, int32_t COL>
	using Matrix = Eigen::Matrix<float32_t, ROW, COL>;

	template<int32_t ROW>
	using Vector = Eigen::Matrix<float32_t, ROW, 1>;
}

#endif // OPENKF_TYPES_H
