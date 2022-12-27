///
/// Copyright 2022 CodingCorner
///
/// Use of this source code is governed by an GPL-3.0 - style
/// license that can be found in the LICENSE file or at
/// https://opensource.org/licenses/GPL-3.0
///
/// @author Mohanad Youssef <mohanad.magdy.hammad@gmail.com>
/// @file square_root_ukf.h
///

#ifndef __SQUARE_ROOT_UNSCENTED_KALMAN_FILTER_LIB_H__
#define __SQUARE_ROOT_UNSCENTED_KALMAN_FILTER_LIB_H__

#include "util.h"
#include "kalman_filter.h"

namespace kf
{
    template<int32_t DIM_X, int32_t DIM_Z>
    class SquareRootUKF : public KalmanFilter<DIM_X, DIM_Z>
    {
    public:
        static constexpr int32_t SIGMA_DIM{ 2 * DIM_X + 1 };

        SquareRootUKF() : KalmanFilter()
        {
            // calculate weights
            const float32_t kappa{ static_cast<float32_t>(3 - DIM_X) };
            updateWeights(kappa);
        }

        ~SquareRootUKF() {}

        Matrix<DIM_X, DIM_X> & matP() override { return (m_matP = m_matSk * m_matSk.transpose()); }
        //const Matrix<DIM_X, DIM_X> & matP() const override { return (m_matP = m_matSk * m_matSk.transpose()); }

        void initialize(const Vector<DIM_X> & vecX, const Matrix<DIM_X, DIM_X> & matP, const Matrix<DIM_X, DIM_X> & matQ, const Matrix<DIM_Z, DIM_Z> & matR)
        {
            m_vecX = vecX;

            {
                // cholesky factorization to get matrix Pk square-root
                Eigen::LLT<Matrix<DIM_X, DIM_X>> lltOfP(matP);
                m_matSk = lltOfP.matrixL(); // sqrt(P)
            }

            {
                // cholesky factorization to get matrix Q square-root
                Eigen::LLT<Matrix<DIM_X, DIM_X>> lltOfRv(matQ);
                m_matRv = lltOfRv.matrixL(); // sqrt(Q)
            }

            {
                // cholesky factorization to get matrix R square-root
                Eigen::LLT<Matrix<DIM_Z, DIM_Z>> lltOfRn(matR);
                m_matRn = lltOfRn.matrixL(); // sqrt(R)
            }
        }

        ///
        /// @brief setting the cholesky factorization of state covariance P.
        /// @param matP state covariance matrix
        ///
        void setCovarianceP(const Matrix<DIM_X, DIM_X> & matP)
        {
            // cholesky factorization to get matrix Q square-root
            Eigen::LLT<Matrix<DIM_X, DIM_X>> lltOfPk(matP);
            m_matSk = lltOfPk.matrixL();
        }

        ///
        /// @brief setting the cholesky factorization of process noise covariance Q.
        /// @param matQ process noise covariance matrix
        ///
        void setCovarianceQ(const Matrix<DIM_X, DIM_X> & matQ)
        {
            // cholesky factorization to get matrix Q square-root
            Eigen::LLT<Matrix<DIM_X, DIM_X>> lltOfRv(matQ);
            m_matRv = lltOfRv.matrixL();
        }

        ///
        /// @brief setting the cholesky factorization of measurement noise covariance R.
        /// @param matR process noise covariance matrix
        ///
        void setCovarianceR(const Matrix<DIM_Z, DIM_Z> & matR)
        {
            // cholesky factorization to get matrix R square-root
            Eigen::LLT<Matrix<DIM_Z, DIM_Z>> lltOfRn(matQ);
            m_matRn = lltOfRn.matrixL(); // sqrt(R)
        }

        ///
        /// @brief state prediction step of the unscented Kalman filter (UKF).
        /// @param predictionModelFunc callback to the prediction/process model function
        ///
        template<typename PredictionModelCallback>
        void predictSRUKF(PredictionModelCallback predictionModelFunc)
        {
            const float32_t kappa{ static_cast<float32_t>(3 - DIM_X) };

            // x_sigmas = self.sigma_points(self.x, self.S)
            Matrix<DIM_X, SIGMA_DIM> matSigmaX{ calculateSigmaPoints(m_vecX, m_matSk, kappa) };

            // y_sigmas = np.zeros((self.dim_x, self.n_sigma))
            // for i in range(self.n_sigma):
            //     y_sigmas[:, i] = f(x_sigmas[:, i])
            for (size_t i{ 0 }; i < SIGMA_DIM; ++i)
            {
                const Vector<DIM_X> Xi{ util::getColumnAt<DIM_X, SIGMA_DIM>(i, matSigmaX) };
                const Vector<DIM_X> Yi{ predictionModelFunc(Xi) }; // y = f(x)

                util::copyToColumn<DIM_X, SIGMA_DIM>(i, matSigmaX, Yi); // Y[:, i] = y
            }

            // calculate weighted mean of the predicted state
            calculateWeightedMean<DIM_X>(matSigmaX, m_vecX);

            // update of cholesky factorized state covariance Sk
            m_matSk = updatePredictedCovariance<DIM_X>(matSigmaX, m_vecX, m_matRv);
        }

        ///
        /// @brief measurement correction step of the unscented Kalman filter (UKF).
        /// @param measurementModelFunc callback to the measurement model function
        /// @param vecZ actual measurement vector.
        ///
        template<typename MeasurementModelCallback>
        void correctSRUKF(MeasurementModelCallback measurementModelFunc, const Vector<DIM_Z> & vecZ)
        {
            const float32_t kappa{ static_cast<float32_t>(3 - DIM_X) };

            // x_sigmas = self.sigma_points(self.x, self.S)
            Matrix<DIM_X, SIGMA_DIM> matSigmaX{ calculateSigmaPoints(m_vecX, m_matSk, kappa) };

            // y_sigmas = np.zeros((self.dim_x, self.n_sigma))
            // for i in range(self.n_sigma):
            //     y_sigmas[:, i] = f(x_sigmas[:, i])
            Matrix<DIM_Z, SIGMA_DIM> matSigmaY;

            for (size_t i{ 0 }; i < SIGMA_DIM; ++i)
            {
                const Vector<DIM_X> Xi{ util::getColumnAt<DIM_X, SIGMA_DIM>(i, matSigmaX) };
                const Vector<DIM_Z> Yi{ measurementModelFunc(Xi) }; // y = f(x)

                util::copyToColumn<DIM_Z, SIGMA_DIM>(i, matSigmaY, Yi); // Y[:, i] = y
            }

            // calculate weighted mean of the predicted state
            Vector<DIM_Z> vecY;
            calculateWeightedMean<DIM_Z>(matSigmaY, vecY);

            // update of cholesky factorized state covariance Sk
            const Matrix<DIM_Z, DIM_Z> matSy{
                updatePredictedCovariance<DIM_Z>(matSigmaY, vecY, m_matRn)
            };

            // cross-correlation
            const Matrix<DIM_X, DIM_Z> matPxy{
                calculateCrossCorrelation(matSigmaX, m_vecX, matSigmaY, vecY)
            };

            // Kalman Gain
            Matrix<DIM_X, DIM_Z> matKk;
            matKk = util::forwardSubstitute<DIM_X, DIM_Z>(matSy, matPxy);
            matKk = util::backwardSubstitute<DIM_X, DIM_Z>(matSy.transpose(), matKk);

            // state vector correction
            m_vecX += matKk * (vecZ - vecY);

            // state covariance correction
            const Matrix<DIM_X, DIM_Z> matU{ matKk * matSy };
            m_matSk = util::cholupdate<DIM_X, DIM_X, DIM_Z>(m_matSk, matU, -1.0);
        }

    private:
        using KalmanFilter<DIM_X, DIM_Z>::m_vecX; // from Base KalmanFilter class
        using KalmanFilter<DIM_X, DIM_Z>::m_matP; // from Base KalmanFilter class

        float32_t m_weight0;  /// @brief unscented transform weight 0 for mean
        float32_t m_weighti;  /// @brief unscented transform weight i for none mean samples

        Matrix<DIM_X, DIM_X> m_matSk{ Matrix<DIM_X, DIM_X>::Zero() };   /// @brief augmented state covariance (incl. process and measurement noise covariances)
        Matrix<DIM_X, DIM_X> m_matRv{ Matrix<DIM_X, DIM_X>::Zero() };   /// @brief augmented state covariance (incl. process and measurement noise covariances)
        Matrix<DIM_Z, DIM_Z> m_matRn{ Matrix<DIM_Z, DIM_Z>::Zero() };   /// @brief augmented state covariance (incl. process and measurement noise covariances)

        ///
        /// @brief algorithm to calculate the weights used to draw the sigma points
        /// @param kappa design scaling parameter for sigma points selection
        ///
        void updateWeights(float32_t kappa)
        {
            static_assert(DIM_X > 0, "DIM_X is Zero which leads to numerical issue.");

            const float32_t denoTerm{ kappa + static_cast<float32_t>(DIM_X) };

            m_weight0 = kappa / denoTerm;
            m_weighti = 0.5F / denoTerm;
        }

        ///
        /// @brief algorithm to calculate the deterministic sigma points for
        /// the unscented transformation
        ///
        /// @param vecXk mean of the normally distributed state
        /// @param matSk covariance of the normally distributed state
        /// @param kappa design scaling parameter for sigma points selection
        ///
        Matrix<DIM_X, SIGMA_DIM> calculateSigmaPoints(const Vector<DIM_X> & vecXk, const Matrix<DIM_X, DIM_X> & matSk, const float32_t kappa)
        {
            const Matrix<DIM_X, DIM_X> matS{ matSk * std::sqrt(DIM_X + kappa) }; // sqrt(n + \kappa) * Sk

            Matrix<DIM_X, SIGMA_DIM> sigmaX;

            // X_0 = \bar{xa}
            util::copyToColumn< DIM_X, SIGMA_DIM >(0, sigmaX, vecXk);

            for (size_t i{ 0 }; i < DIM_X; ++i)
            {
                const size_t IDX_1{ i + 1 };
                const size_t IDX_2{ i + DIM_X + 1 };

                util::copyToColumn< DIM_X, SIGMA_DIM >(IDX_1, sigmaX, vecXk);
                util::copyToColumn< DIM_X, SIGMA_DIM >(IDX_2, sigmaX, vecXk);

                const Vector<DIM_X> vecShiftTerm{ util::getColumnAt< DIM_X, DIM_X >(i, matS) };

                util::addColumnFrom< DIM_X, SIGMA_DIM >(IDX_1, sigmaX, vecShiftTerm);  // X_i     = \bar{x} + sqrt(n + \kappa) * Sk
                util::subColumnFrom< DIM_X, SIGMA_DIM >(IDX_2, sigmaX, vecShiftTerm);  // X_{i+n} = \bar{x} - sqrt(n + \kappa) * Sk
            }

            return sigmaX;
        }

        ///
        /// @brief calculate the weighted mean and covariance given a set of sigma points
        /// @param sigmaX matrix of sigma points where each column contain single sigma point
        /// @param vecX output weighted mean
        ///
        template<size_t DIM>
        void calculateWeightedMean(const Matrix<DIM, SIGMA_DIM> & sigmaX, Vector<DIM> & vecX)
        {
            // 1. calculate mean: \bar{y} = \sum_{i_0}^{2n} W[0, i] Y[:, i]
            vecX = m_weight0 * util::getColumnAt<DIM, SIGMA_DIM>(0, sigmaX);
            for (size_t i{ 1 }; i < SIGMA_DIM; ++i)
            {
                vecX += m_weighti * util::getColumnAt<DIM, SIGMA_DIM>(i, sigmaX); // y += W[0, i] Y[:, i]
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

            for (size_t i{ 1 }; i < SIGMA_DIM; ++i)
            {
                devXi = util::getColumnAt<DIM_X, SIGMA_DIM>(i, sigmaX) - vecX; // X[:, i] - \bar{x}
                devYi = util::getColumnAt<DIM_Z, SIGMA_DIM>(i, sigmaY) - vecY; // Y[:, i] - \bar{y}

                matPxy += m_weighti * (devXi * devYi.transpose()); // y += W[0, i] (Y[:, i] - \bar{y}) (Y[:, i] - \bar{y})^T
            }

            return matPxy;
        }

        template<int32_t DIM>
        Matrix<DIM, DIM> updatePredictedCovariance(
            const Matrix<DIM, SIGMA_DIM> & matSigmaX, const Vector<DIM> & meanX, const Matrix<DIM, DIM> & matU)
        {
            constexpr int32_t DIM_ROW{ SIGMA_DIM + DIM - 1 };
            constexpr int32_t DIM_COL{ DIM };

            // build compound matrix for square - root covariance update
            // sigmas_X[:, 0] is not added because W0 could be zero which will lead
            // to undefined outcome from sqrt(W0).
            Matrix<DIM_ROW, DIM_COL> matC{ buildCompoundMatrix(matSigmaX, meanX, matU) };

            // calculate square - root covariance S using QR decomposition of compound matrix C
            // including the process noise covariance
            // _, S_minus = np.linalg.qr(C)
            Eigen::HouseholderQR< Matrix<DIM_ROW, DIM_COL> > qr(matC);

            // get the upper triangular matrix from the factorization
            // might need to reimplement QR factorization it as it seems to use dynamic memory allocation
            Matrix<DIM, DIM> matR{
                util::getBlock<DIM_ROW, DIM_COL, DIM, DIM>(
                    qr.matrixQR().triangularView<Eigen::Upper>(), 0, 0)
            };

            // Rank - 1 cholesky update
            // x_dev = sigmas_X[:, 0] - x_minus
            // x_dev = np.reshape(x_dev, [-1, 1])
            // S_minus = cholupdate(S_minus.T, x_dev, self.W0)
            // TODO: implement cholupdate, the one from Eigen is not straight forward to use
            Vector<DIM> devX0 = util::getColumnAt<DIM, SIGMA_DIM>(0, matSigmaX) - meanX;
            matR = util::cholupdate<DIM, DIM, 1>(matR.transpose(), devX0, m_weight0);

            return matR;
        }

        Matrix<SIGMA_DIM + DIM_X - 1, DIM_X>
        buildCompoundMatrix(
            const Matrix<DIM_X, SIGMA_DIM> & matSigmaX, const Vector<DIM_X> & meanX, const Matrix<DIM_X, DIM_X> & matU)
        {
            // build compoint/joint matrix for square-root covariance update

            constexpr int32_t DIM_ROW{ SIGMA_DIM + DIM_X - 1 };
            constexpr int32_t DIM_COL{ DIM_X };

            Matrix<DIM_ROW, DIM_COL> matC;

            // C = (sigmas_X[:, 1 : ].T - x_minus) * np.sqrt(self.Wi)
            const float32_t sqrtWi{ std::sqrt(m_weighti) };
            for (int32_t i{ 0 }; i < DIM_X; ++i)
            {
                for (int32_t j{ 1 }; j < SIGMA_DIM; ++j)
                {
                    matC(j - 1, i) = sqrtWi * (matSigmaX(i, j) - meanX[i]);
                }
            }

            // C = np.concatenate((C, self.sqrt_Q.T), axis=0)
            for (int32_t i{ 0 }; i < DIM_X; ++i)
            {
                for (int32_t j{ 0 }; j < DIM_X; ++j)
                {
                    matC(j + SIGMA_DIM - 1, i) = matU(i, j);
                }
            }

            return matC;
        }
    };
}

#endif // __UNSCENTED_KALMAN_FILTER_LIB_H__
