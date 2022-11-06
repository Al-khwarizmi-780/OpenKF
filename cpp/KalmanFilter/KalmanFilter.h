///
/// @author Mohanad Youssef
/// @file KalmanFilterExercise/KalmanFilter/KalmanFilter.h
///

#ifndef __KALMAN_FILTER_LIB_H__
#define __KALMAN_FILTER_LIB_H__

#include "Types.h"

namespace kf
{
    template<size_t DIM_X, size_t DIM_Z>
    class KalmanFilter
    {
    public:

        KalmanFilter()
        {

        }

        ~KalmanFilter()
        {

        }

        Vector<DIM_X> & vecX() { return m_vecX; }
        const Vector<DIM_X> & vecX() const { return m_vecX; }

        Matrix<DIM_X, DIM_X> & matP() { return m_matP; }
        const Matrix<DIM_X, DIM_X> & matP() const { return m_matP; }

        void predict(const Matrix<DIM_X, DIM_X> & matF, const Matrix<DIM_X, DIM_X> & matQ)
        {
            m_vecX = matF * m_vecX;
            m_matP = matF * m_matP * matF.transpose() + matQ;
        }

        void correct(const Vector<DIM_Z> & vecZ, const Matrix<DIM_Z, DIM_Z> & matR, const Matrix<DIM_Z, DIM_X> & matH)
        {
            Matrix<DIM_X, DIM_X> matI;
            matI.setIdentity();

            Matrix<DIM_Z, DIM_Z> matSk; // Innovation covariance
            matSk = matH * m_matP * matH.transpose() + matR;

            Matrix<DIM_X, DIM_Z> matKk; // Kalman Gain
            matKk = m_matP * matH.transpose() * matSk.inverse();

            m_vecX = m_vecX + matKk * (vecZ - (matH * m_vecX));
            m_matP = (matI - matKk * matH) * m_matP;
        }

    private:
        Vector<DIM_X> m_vecX; /// @brief estimated state vector
        Matrix<DIM_X, DIM_X> m_matP; /// @brief state covariance matrix
    };
}

#endif // __KALMAN_FILTER_LIB_H__