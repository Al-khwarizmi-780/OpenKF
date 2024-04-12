///
/// Copyright 2022 Mohanad Youssef (Al-khwarizmi)
///
/// Use of this source code is governed by an GPL-3.0 - style
/// license that can be found in the LICENSE file or at
/// https://opensource.org/licenses/GPL-3.0
///
/// @author Mohanad Youssef <mohanad.magdy.hammad@gmail.com>
/// @file unscented_kalman_filter.h
///

#ifndef UNSCENTED_KALMAN_FILTER_LIB_H
#define UNSCENTED_KALMAN_FILTER_LIB_H

#include "util.h"
#include "kalman_filter.h"

namespace kf
{
    template<int32_t DIM_X, int32_t DIM_Z, int32_t DIM_V, int32_t DIM_N>
    class UnscentedKalmanFilter : public KalmanFilter<DIM_X, DIM_Z>
    {
    public:
        static constexpr int32_t DIM_A{ DIM_X + DIM_V + DIM_N };
        static constexpr int32_t SIGMA_DIM{ 2 * DIM_A + 1 };

        UnscentedKalmanFilter()
            : KalmanFilter<DIM_X, DIM_Z>()
        {
            // 1. calculate weights
            const float32_t kappa{ static_cast<float32_t>(3 - DIM_A) };
            updateWeights(kappa);
        }

        ~UnscentedKalmanFilter() {}

        ///
        /// @brief adding process noise covariance Q to the augmented state covariance matPa
        /// in the middle element of the diagonal.
        ///
        void setCovarianceQ(const Matrix<DIM_V, DIM_V> & matQ)
        {
            const int32_t S_IDX{ DIM_X };
            const int32_t L_IDX{ S_IDX + DIM_V };

            for (int32_t i{ S_IDX }; i < L_IDX; ++i)
            {
                for (int32_t j{ S_IDX }; j < L_IDX; ++j)
                {
                    m_matPa(i, j) = matQ(i - S_IDX, j - S_IDX);
                }
            }
        }

        ///
        /// @brief adding measurement noise covariance R to the augmented state covariance matPa
        /// in the third element of the diagonal.
        ///
        void setCovarianceR(const Matrix<DIM_N, DIM_N> & matR)
        {
            const int32_t S_IDX{ DIM_X + DIM_V };
            const int32_t L_IDX{ S_IDX + DIM_N };

            for (int32_t i{ S_IDX }; i < L_IDX; ++i)
            {
                for (int32_t j{ S_IDX }; j < L_IDX; ++j)
                {
                    m_matPa(i, j) = matR(i - S_IDX, j - S_IDX);
                }
            }
        }

        ///
        /// @brief adding vecX and matP to the augmented state vector and covariance vecXa and matPa
        ///
        void updateAugmentedStateAndCovariance()
        {
            updateAugmentedVecX();
            updateAugmentedMatP();
        }

        ///
        /// @brief state prediction step of the unscented Kalman filter (UKF).
        /// @param predictionModelFunc callback to the prediction/process model function
        ///
        template<typename PredictionModelCallback>
        void predictUKF(PredictionModelCallback predictionModelFunc)
        {
            // self.x_a[:self.dim_x] = x
            // self.P_a[:self.dim_x, : self.dim_x] = P
            updateAugmentedStateAndCovariance();

            const float32_t kappa{ static_cast<float32_t>(3 - DIM_A) };

            // xa_sigmas = self.sigma_points(self.x_a, self.P_a)
            Matrix<DIM_A, SIGMA_DIM> matSigmaXa{ calculateSigmaPoints(m_vecXa, m_matPa, kappa) };

            // xx_sigmas = xa_sigmas[:self.dim_x, :]
            //Matrix<DIM_X, SIGMA_DIM> sigmaXx{ matSigmaXa.block<DIM_X, SIGMA_DIM>(0, 0) };
            Matrix<DIM_X, SIGMA_DIM> sigmaXx{ matSigmaXa.block(0, 0, DIM_X, SIGMA_DIM) };

            // xv_sigmas = xa_sigmas[self.dim_x : self.dim_x + self.dim_v, :]
            Matrix<DIM_V, SIGMA_DIM> sigmaXv{ matSigmaXa.block(DIM_X, 0, DIM_V, SIGMA_DIM) };

            // y_sigmas = np.zeros((self.dim_x, self.n_sigma))
            // for i in range(self.n_sigma):
            //     y_sigmas[:, i] = f(xx_sigmas[:, i], xv_sigmas[:, i])
            for (int32_t i{ 0 }; i < SIGMA_DIM; ++i)
            {
                const Vector<DIM_X> sigmaXxi{ util::getColumnAt<DIM_X, SIGMA_DIM>(i, sigmaXx) };
                const Vector<DIM_V> sigmaXvi{ util::getColumnAt<DIM_V, SIGMA_DIM>(i, sigmaXv) };

                const Vector<DIM_X> Yi{ predictionModelFunc(sigmaXxi, sigmaXvi) }; // y = f(x)

                util::copyToColumn<DIM_X, SIGMA_DIM>(i, sigmaXx, Yi); // Y[:, i] = y
            }

            // y, Pyy = self.calculate_mean_and_covariance(y_sigmas)
            calculateWeightedMeanAndCovariance<DIM_X>(sigmaXx, m_vecX, m_matP);

            //// self.x_a[:self.dim_x] = y
            //// self.P_a[:self.dim_x, : self.dim_x] = Pyy
            //updateAugmentedStateAndCovariance();
        }

        ///
        /// @brief measurement correction step of the unscented Kalman filter (UKF).
        /// @param measurementModelFunc callback to the measurement model function
        /// @param vecZ actual measurement vector.
        ///
        template<typename MeasurementModelCallback>
        void correctUKF(MeasurementModelCallback measurementModelFunc, const Vector<DIM_Z> & vecZ)
        {
            // self.x_a[:self.dim_x] = x
            // self.P_a[:self.dim_x, : self.dim_x] = P
            updateAugmentedStateAndCovariance();

            const float32_t kappa{ static_cast<float32_t>(3 - DIM_A) };

            // xa_sigmas = self.sigma_points(self.x_a, self.P_a)
            Matrix<DIM_A, SIGMA_DIM> matSigmaXa{ calculateSigmaPoints(m_vecXa, m_matPa, kappa) };

            // xx_sigmas = xa_sigmas[:self.dim_x, :]
            Matrix<DIM_X, SIGMA_DIM> sigmaXx{ matSigmaXa.block(0, 0, DIM_X, SIGMA_DIM) };

            // xn_sigmas = xa_sigmas[self.dim_x + self.dim_v :, :]
            Matrix<DIM_N, SIGMA_DIM> sigmaXn{ matSigmaXa.block(DIM_X + DIM_V, 0, DIM_N, SIGMA_DIM) };

            // y_sigmas = np.zeros((self.dim_z, self.n_sigma))
            // for i in range(self.n_sigma) :
            //     y_sigmas[:, i] = h(xx_sigmas[:, i], xn_sigmas[:, i])
            Matrix<DIM_Z, SIGMA_DIM> sigmaY;
            for (int32_t i{ 0 }; i < SIGMA_DIM; ++i)
            {
                const Vector<DIM_X> sigmaXxi{ util::getColumnAt<DIM_X, SIGMA_DIM>(i, sigmaXx) };
                const Vector<DIM_N> sigmaXni{ util::getColumnAt<DIM_N, SIGMA_DIM>(i, sigmaXn) };

                const Vector<DIM_Z> Yi{ measurementModelFunc(sigmaXxi, sigmaXni) }; // y = f(x)

                util::copyToColumn<DIM_Z, SIGMA_DIM>(i, sigmaY, Yi); // Y[:, i] = y
            }

            // y, Pyy = self.calculate_mean_and_covariance(y_sigmas)
            Vector<DIM_Z> vecY;
            Matrix<DIM_Z, DIM_Z> matPyy;
            calculateWeightedMeanAndCovariance<DIM_Z>(sigmaY, vecY, matPyy);

            // TODO: calculate cross correlation
            const Matrix<DIM_X, DIM_Z> matPxy{ calculateCrossCorrelation(sigmaXx, m_vecX, sigmaY, vecY) };

            // kalman gain
            const Matrix<DIM_X, DIM_Z> matK{ matPxy * matPyy.inverse() };

            m_vecX += matK * (vecZ - vecY);
            m_matP -= matK * matPyy * matK.transpose();

            //// self.x_a[:self.dim_x] = x
            //// self.P_a[:self.dim_x, : self.dim_x] = P
            //updateAugmentedStateAndCovariance();
        }

    private:
        using KalmanFilter<DIM_X, DIM_Z>::m_vecX; // from Base KalmanFilter class
        using KalmanFilter<DIM_X, DIM_Z>::m_matP; // from Base KalmanFilter class

        float32_t m_weight0;  /// @brief unscented transform weight 0 for mean
        float32_t m_weighti;  /// @brief unscented transform weight i for none mean samples

        Vector<DIM_A> m_vecXa{ Vector<DIM_A>::Zero() };                 /// @brief augmented state vector (incl. process and measurement noise means)
        Matrix<DIM_A, DIM_A> m_matPa{ Matrix<DIM_A, DIM_A>::Zero() };   /// @brief augmented state covariance (incl. process and measurement noise covariances)

        ///
        /// @brief add state vector m_vecX to the augment state vector m_vecXa
        ///
        void updateAugmentedVecX()
        {
            for (int32_t i{ 0 }; i < DIM_X; ++i)
            {
                m_vecXa[i] = m_vecX[i];
            }
        }

        ///
        /// @brief add covariance matrix m_matP to the augment covariance m_matPa
        ///
        void updateAugmentedMatP()
        {
            for (int32_t i{ 0 }; i < DIM_X; ++i)
            {
                for (int32_t j{ 0 }; j < DIM_X; ++j)
                {
                    m_matPa(i, j) = m_matP(i, j);
                }
            }
        }

        ///
        /// @brief algorithm to calculate the weights used to draw the sigma points
        /// @param kappa design scaling parameter for sigma points selection
        ///
        void updateWeights(float32_t kappa)
        {
            static_assert(DIM_A > 0, "DIM_A is Zero which leads to numerical issue.");

            const float32_t denoTerm{ kappa + static_cast<float32_t>(DIM_A) };

            m_weight0 = kappa / denoTerm;
            m_weighti = 0.5F / denoTerm;
        }

        ///
        /// @brief algorithm to calculate the deterministic sigma points for
        /// the unscented transformation
        ///
        /// @param vecX mean of the normally distributed state
        /// @param matPxx covariance of the normally distributed state
        /// @param kappa design scaling parameter for sigma points selection
        ///
        Matrix<DIM_A, SIGMA_DIM> calculateSigmaPoints(const Vector<DIM_A> & vecXa, const Matrix<DIM_A, DIM_A> & matPa, const float32_t kappa)
        {
            const float32_t scalarMultiplier{ std::sqrt(DIM_A + kappa) }; // sqrt(n + \kappa)

            // cholesky factorization to get matrix Pxx square-root
            Eigen::LLT<Matrix<DIM_A, DIM_A>> lltOfPa(matPa);
            Matrix<DIM_A, DIM_A> matSa{ lltOfPa.matrixL() }; // sqrt(P_{a})

            matSa *= scalarMultiplier; // sqrt( (n + \kappa) * P_{a} )

            Matrix<DIM_A, SIGMA_DIM> sigmaXa;

            // X_0 = \bar{xa}
            util::copyToColumn< DIM_A, SIGMA_DIM >(0, sigmaXa, vecXa);

            for (int32_t i{ 0 }; i < DIM_A; ++i)
            {
                const int32_t IDX_1{ i + 1 };
                const int32_t IDX_2{ i + DIM_A + 1 };

                util::copyToColumn< DIM_A, SIGMA_DIM >(IDX_1, sigmaXa, vecXa);
                util::copyToColumn< DIM_A, SIGMA_DIM >(IDX_2, sigmaXa, vecXa);

                const Vector<DIM_A> vecShiftTerm{ util::getColumnAt< DIM_A, DIM_A >(i, matSa) };

                util::addColumnFrom< DIM_A, SIGMA_DIM >(IDX_1, sigmaXa, vecShiftTerm);  // X_i^a     = \bar{xa} + sqrt( (n^a + \kappa) * P^{a} )
                util::subColumnFrom< DIM_A, SIGMA_DIM >(IDX_2, sigmaXa, vecShiftTerm);  // X_{i+n}^a = \bar{xa} - sqrt( (n^a + \kappa) * P^{a} )
            }

            return sigmaXa;
        }

        ///
        /// @brief calculate the weighted mean and covariance given a set of sigma points
        /// @param sigmaX matrix of sigma points where each column contain single sigma point
        /// @param vecX output weighted mean
        /// @param matP output weighted covariance
        ///
        template<int32_t STATE_DIM>
        void calculateWeightedMeanAndCovariance(const Matrix<STATE_DIM, SIGMA_DIM> & sigmaX, Vector<STATE_DIM> & vecX, Matrix<STATE_DIM, STATE_DIM> & matPxx)
        {
            // 1. calculate mean: \bar{y} = \sum_{i_0}^{2n} W[0, i] Y[:, i]
            vecX = m_weight0 * util::getColumnAt<STATE_DIM, SIGMA_DIM>(0, sigmaX);
            for (int32_t i{ 1 }; i < SIGMA_DIM; ++i)
            {
                vecX += m_weighti * util::getColumnAt<STATE_DIM, SIGMA_DIM>(i, sigmaX); // y += W[0, i] Y[:, i]
            }

            // 2. calculate covariance: P_{yy} = \sum_{i_0}^{2n} W[0, i] (Y[:, i] - \bar{y}) (Y[:, i] - \bar{y})^T
            Vector<STATE_DIM> devXi{ util::getColumnAt<STATE_DIM, SIGMA_DIM>(0, sigmaX) - vecX }; // Y[:, 0] - \bar{ y }
            matPxx = m_weight0 * devXi * devXi.transpose(); // P_0 = W[0, 0] (Y[:, 0] - \bar{y}) (Y[:, 0] - \bar{y})^T

            for (int32_t i{ 1 }; i < SIGMA_DIM; ++i)
            {
                devXi = util::getColumnAt<STATE_DIM, SIGMA_DIM>(i, sigmaX) - vecX; // Y[:, i] - \bar{y}

                const Matrix<STATE_DIM, STATE_DIM> Pi{ m_weighti * devXi * devXi.transpose() }; // P_i = W[0, i] (Y[:, i] - \bar{y}) (Y[:, i] - \bar{y})^T

                matPxx += Pi; // y += W[0, i] (Y[:, i] - \bar{y}) (Y[:, i] - \bar{y})^T
            }
        }

        ///
        /// @brief calculate the cross-correlation given two sets sigma points X and Y and their means x and y
        /// @param sigmaX first matrix of sigma points where each column contain single sigma point
        /// @param vecX mean of the first set of sigma points
        /// @param sigmaY second matrix of sigma points where each column contain single sigma point
        /// @param vecY mean of the second set of sigma points
        /// @return matPxy, the cross-correlation matrix
        ///
        Matrix<DIM_X, DIM_Z> calculateCrossCorrelation(
            const Matrix<DIM_X, SIGMA_DIM> & sigmaX, const Vector<DIM_X> & vecX,
            const Matrix<DIM_Z, SIGMA_DIM> & sigmaY, const Vector<DIM_Z> & vecY)
        {
            Vector<DIM_X> devXi{ util::getColumnAt<DIM_X, SIGMA_DIM>(0, sigmaX) - vecX }; // X[:, 0] - \bar{ x }
            Vector<DIM_Z> devYi{ util::getColumnAt<DIM_Z, SIGMA_DIM>(0, sigmaY) - vecY }; // Y[:, 0] - \bar{ y }

            // P_0 = W[0, 0] (X[:, 0] - \bar{x}) (Y[:, 0] - \bar{y})^T
            Matrix<DIM_X, DIM_Z> matPxy{
                m_weight0 * (devXi * devYi.transpose())
            }; 

            for (int32_t i{ 1 }; i < SIGMA_DIM; ++i)
            {
                devXi = util::getColumnAt<DIM_X, SIGMA_DIM>(i, sigmaX) - vecX; // X[:, i] - \bar{x}
                devYi = util::getColumnAt<DIM_Z, SIGMA_DIM>(i, sigmaY) - vecY; // Y[:, i] - \bar{y}

                matPxy += m_weighti * (devXi * devYi.transpose()); // y += W[0, i] (Y[:, i] - \bar{y}) (Y[:, i] - \bar{y})^T
            }

            return matPxy;
        }
    };
}

#endif // UNSCENTED_KALMAN_FILTER_LIB_H
