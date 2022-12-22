///
/// Copyright 2022 CodingCorner
///
/// Use of this source code is governed by an GPL-3.0 - style
/// license that can be found in the LICENSE file or at
/// https://opensource.org/licenses/GPL-3.0
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

    template<size_t ROWS, size_t COLS, size_t N_ROWS, size_t N_COLS>
    Matrix<N_ROWS, N_COLS> getBlock(const Matrix<ROWS, COLS> & matA, int32_t startRowIdx, int32_t startColIdx)
    {
        Matrix<N_ROWS, N_COLS> matB;

        for (int32_t i = startRowIdx; i < startRowIdx + N_ROWS; ++i)
        {
            for (int32_t j = startColIdx; j < startColIdx + N_COLS; ++j)
            {
                matB(i - startRowIdx, j - startColIdx) = matA(i, j);
            }
        }

        return matB;
    }

    template<size_t ROWS, size_t COLS_L, size_t COLS_W>
    Matrix<ROWS, COLS_L> cholupdate(Matrix<ROWS, COLS_L> matL, Matrix<ROWS, COLS_W> matW, float32_t alpha)
    {
        Matrix<ROWS, COLS_L> matLo;

        for (int32_t i{ 0 }; i < COLS_W; ++i)
        {
            matLo = matL;

            float32_t b{ 1.0F };

            for (int32_t j{ 0 }; j < ROWS; ++j)
            {
                const float32_t ljjPow2{ matL(j, j) * matL(j, j) };
                const float32_t wjiPow2{ matW(j, i) * matW(j, i) };
                const float32_t upsilon{ (ljjPow2 * b) + (alpha * wjiPow2) };

                matLo(j, j) = std::sqrt(ljjPow2 + ((alpha / b) * wjiPow2));

                for (int32_t k{ j + 1 }; k < ROWS; ++k)
                {
                    matW(k, i) -= (matW(j, i) / matL(j, j)) * matL(k, j);
                    matLo(k, j) = ((matLo(j, j) / matL(j, j)) * matL(k, j)) + (matLo(j, j) * alpha * matW(j, i) * matW(k, i) / upsilon);
                }

                b += alpha * (wjiPow2 / ljjPow2);
            }

            matL = matLo;
        }
        return matLo;
    }

    template<size_t ROWS, size_t COLS>
    Matrix<ROWS, COLS> forwardSubstitute(const Matrix<ROWS, ROWS> & matA, const Matrix<ROWS, COLS> & matB)
    {
        Matrix<ROWS, COLS> matX;

        for (int32_t k{ 0 }; k < COLS; ++k)
        {
            for (int32_t i{ 0 }; i < ROWS; ++i)
            {
                float32_t accumulation{ matB(i, k) };
                for (int32_t j{ 0 }; j < i; ++j)
                {
                    accumulation -= matA(i, j) * matX(j, k);
                }

                matX(i, k) = accumulation / matA(i, i);
            }
        }
        return matX;
    }

    template<size_t ROWS, size_t COLS>
    Matrix<ROWS, COLS> backwardSubstitute(const Matrix<ROWS, ROWS> & matA, const Matrix<ROWS, COLS> & matB)
    {
        Matrix<ROWS, COLS> matX;

        for (int32_t k{ 0 }; k < COLS; ++k)
        {
            for (int32_t i{ ROWS - 1 }; i >= 0; --i)
            {
                float32_t accumulation{ matB(i, k) };

                for (int32_t j{ ROWS - 1 }; j > i; --j)
                {
                    accumulation -= matA(i, j) * matX(j, k);
                }

                matX(i, k) = accumulation / matA(i, i);
            }
        }
        return matX;
    }

    template<size_t ROWS1, size_t ROWS2, size_t COLS>
    class JointRows
    {
    public:
        JointRows() = delete;
        ~JointRows() {}

        explicit JointRows(const Matrix<ROWS1, COLS> & matM1, const Matrix<ROWS2, COLS> & matM2)
        {
            for (int32_t j{ 0 }; j < COLS; ++j)
            {
                for (int32_t i{ 0 }; i < ROWS1; ++i)
                {
                    m_matJ(i, j) = matM1(i, j);
                }

                for (int32_t i{ 0 }; i < ROWS2; ++i)
                {
                    m_matJ(i + ROWS1, j) = matM2(i, j);
                }
            }
        }

        const Matrix<ROWS1 + ROWS2, COLS> & jointMatrix() const { return m_matJ; }

    private:
        Matrix<ROWS1 + ROWS2, COLS> m_matJ;
    };

    template<size_t ROWS, size_t COLS1, size_t COLS2>
    class JointCols
    {
    public:
        JointCols() = delete;
        ~JointCols() {}

        explicit JointCols(const Matrix<ROWS, COLS1> & matM1, const Matrix<ROWS, COLS2> & matM2)
        {
            for (int32_t i{ 0 }; i < ROWS; ++i)
            {
                for (int32_t j{ 0 }; j < COLS1; ++j)
                {
                    m_matJ(i, j) = matM1(i, j);
                }

                for (int32_t j{ 0 }; j < COLS2; ++j)
                {
                    m_matJ(i, j + COLS1) = matM2(i, j);
                }
            }
        }

        const Matrix<ROWS, COLS1 + COLS2> & jointMatrix() const { return m_matJ; }

    private:
        Matrix<ROWS, COLS1 + COLS2> m_matJ;
    };
}

#endif // __KALMAN_FILTER_UTIL_H__
