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

#include <Eigen/Dense>
#include <stdint.h>

namespace kf
{
using float32_t = float;

template <int32_t ROW, int32_t COL>
using Matrix = Eigen::Matrix<float32_t, ROW, COL>;

template <int32_t ROW>
using Vector = Eigen::Matrix<float32_t, ROW, 1>;

struct PointXY
{
  PointXY() : x{0.0}, y{0.0} {}
  PointXY(double const _x, double const _y) : x{_x}, y{_y} {}

  PointXY operator+(PointXY const& p2) const
  {
    return PointXY(x + p2.x, y + p2.y);
  }
  PointXY operator-(PointXY const& p2) const
  {
    return PointXY(x - p2.x, y - p2.y);
  }
  double operator*(PointXY const& p2) const { return x * p2.x + y * p2.y; }

  double x;
  double y;
};
using VecXY = PointXY;

struct Dimension
{
  Dimension() : width{0.0}, length{0.0} {}
  Dimension(double const w, double const l) : width{w}, length{l} {}
  double width;
  double length;
};

struct Rectangle
{
  PointXY center;
  Dimension dimension;
  double angle;
};

}  // namespace kf

#endif  // OPENKF_TYPES_H
