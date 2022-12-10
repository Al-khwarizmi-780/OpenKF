///
/// Copyright 2022 CodingCorner
///
/// Use of this source code is governed by an GPL - style
/// license that can be found in the LICENSE file or at
/// https ://https://opensource.org/licenses/GPL-3.0.
///
/// @author Mohanad Youssef <mohanad.magdy.hammad@gmail.com>
/// @file util.h
///

#ifndef __KALMAN_FILTER_UTIL_H__
#define __KALMAN_FILTER_UTIL_H__

#include "types.h"

namespace util
{
    template<size_t ROWS, size_t COLS>
    void copyToColumn(const size_t colIdx, Matrix<ROWS, COLS> & lhsSigmaX, const Vector<ROWS> & rhsVecX)
    {
        for (size_t i{ 0 }; i < ROWS; ++i)
        { // rows
            lhsSigmaX(i, colIdx) = rhsVecX[i];
        }
    }

    template<size_t ROWS, size_t COLS>
    void addColumnFrom(const size_t colIdx, Matrix<ROWS, COLS> & lhsSigmaX, const Vector<ROWS> & rhsVecX)
    {
        for (size_t i{ 0 }; i < ROWS; ++i)
        { // rows
            lhsSigmaX(i, colIdx) += rhsVecX[i];
        }
    }

    template<size_t ROWS, size_t COLS>
    void subColumnFrom(const size_t colIdx, Matrix<ROWS, COLS> & lhsSigmaX, const Vector<ROWS> & rhsVecX)
    {
        for (size_t i{ 0 }; i < ROWS; ++i)
        { // rows
            lhsSigmaX(i, colIdx) -= rhsVecX[i];
        }
    }

    template<size_t ROWS, size_t COLS>
    Vector<ROWS> getColumnAt(const size_t colIdx, const Matrix<ROWS, COLS> & matX)
    {
        assert(colIdx < COLS); // assert if colIdx is out of boundary

        Vector<ROWS> vecXi;

        for (size_t i{ 0 }; i < ROWS; ++i)
        { // rows
            vecXi[i] = matX(i, colIdx);
        }

        return vecXi;
    }
}

#endif // __KALMAN_FILTER_UTIL_H__
