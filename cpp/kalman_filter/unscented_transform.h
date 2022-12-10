///
/// Copyright 2022 CodingCorner
///
/// Use of this source code is governed by an GPL - style
/// license that can be found in the LICENSE file or at
/// https ://https://opensource.org/licenses/GPL-3.0.
///
/// @author Mohanad Youssef <mohanad.magdy.hammad@gmail.com>
/// @file unscented_transform.h
///

#ifndef __UNSCENTED_TRANSFORM_H__
#define __UNSCENTED_TRANSFORM_H__

#include "types.h"
#include "util.h"
#include "Eigen/src/Cholesky/LLT.h"

namespace kf
{
    template<size_t DIM>
    class UnscentedTransform
    {
    public:
        static constexpr size_t SIGMA_DIM{ (2 * DIM) + 1 };

        UnscentedTransform()
        {
            // 1. calculate weights
            updateWeights(kappa);
        }

        ~UnscentedTransform() {}

        float32_t weight0() const { return _weight0; }
        float32_t weighti() const { return _weighti; }

        ///
        /// @brief algorithm to execute weight and sigma points calculation
        /// @param vecX input mean vector
        /// @param matPxx input covariance matrix
        /// @param kappa design parameter
        ///
        void compute(const Vector<DIM> & vecX, const Matrix<DIM, DIM> & matPxx, const float32_t kappa = 0.0F)
        {
            // 2. update sigma points _sigmaX
            updateSigmaPoints(vecX, matPxx, kappa);
        }

        template<size_t DIM_X>
        void calculateWeightedMeanAndCovariance(const Matrix<DIM_X, SIGMA_DIM> & sigmaX, Vector<DIM_X> & vecX, Matrix<DIM_X, DIM_X> & matPxx)
        {
            // 1. calculate mean: \bar{y} = \sum_{i_0}^{2n} W[0, i] Y[:, i]
            vecX = _weight0 * util::getColumnAt<DIM_X, SIGMA_DIM>(0, sigmaX);
            for (size_t i{ 1 }; i < SIGMA_DIM; ++i)
            {
                vecX += _weighti * util::getColumnAt<DIM_X, SIGMA_DIM>(i, sigmaX); // y += W[0, i] Y[:, i]
            }

            // 2. calculate covariance: P_{yy} = \sum_{i_0}^{2n} W[0, i] (Y[:, i] - \bar{y}) (Y[:, i] - \bar{y})^T
            Vector<DIM_X> devXi{ util::getColumnAt<DIM_X, SIGMA_DIM>(0, sigmaX) - vecX }; // Y[:, 0] - \bar{ y }
            matPxx = _weight0 * devXi * devXi.transpose(); // P_0 = W[0, 0] (Y[:, 0] - \bar{y}) (Y[:, 0] - \bar{y})^T

            for (size_t i{ 1 }; i < SIGMA_DIM; ++i)
            {
                devXi = util::getColumnAt<DIM_X, SIGMA_DIM>(i, sigmaX) - vecX; // Y[:, i] - \bar{y}

                const Matrix<DIM_X, DIM_X> Pi{ _weighti * devXi * devXi.transpose() }; // P_i = W[0, i] (Y[:, i] - \bar{y}) (Y[:, i] - \bar{y})^T

                matPxx += Pi; // y += W[0, i] (Y[:, i] - \bar{y}) (Y[:, i] - \bar{y})^T
            }
        }

        ///
        /// @brief algorithm to execute transforming sigma points through nonlinear function
        /// and calculate the updated weights and covariance.
        /// @param nonlinearFunction nonlinear function callback
        /// @param vecY output mean vector (weighted mean vector)
        /// @param matPyy output covariance matrix (weighted covariance matrix)
        ///
        template<typename NonLinearFunctionCallback>
        void transform(NonLinearFunctionCallback nonlinearFunction, Vector<DIM> & vecY, Matrix<DIM, DIM> & matPyy)
        {
            Matrix<DIM, SIGMA_DIM> sigmaY;

            // 3. transform sigma points X through the nonlinear model f(X) to obtain Y sigma points
            transformSigmaPoints(nonlinearFunction, sigmaY);

            // 4. calculate weight mean and covariance from the transformed sigma points Y and weights
            updateTransformedMeanAndCovariance(sigmaY, vecY, matPyy);
        }

        void showSummary()
        {
            std::cout << "DIM_N : " << DIM << "\n";
            std::cout << "DIM_M : " << SIGMA_DIM << "\n";
            std::cout << "_weight0 : \n" << _weight0 << "\n";
            std::cout << "_weighti : \n" << _weighti << "\n";
            std::cout << "_sigmaX : \n" << _sigmaX << "\n";
        }

        Matrix<1, SIGMA_DIM> & weights() { return _weights; }
        const Matrix<1, SIGMA_DIM> & weights() const { return _weights; }

        Matrix<DIM, SIGMA_DIM> & sigmaX() { return _sigmaX; }
        const Matrix<DIM, SIGMA_DIM> & sigmaX() const { return _sigmaX; }

    private:
        float32_t _weight0;  /// @brief unscented transform weight 0 for mean
        float32_t _weighti;  /// @brief unscented transform weight 0 for none mean samples

        Matrix<DIM, SIGMA_DIM> _sigmaX; /// @brief input sigma points

        ///
        /// @brief algorithm to calculate the weights used to draw the sigma points
        /// @param kappa design scaling parameter for sigma points selection
        ///
        void updateWeights(float32_t kappa)
        {
            static_assert(DIM > 0, "DIM is Zero which leads to numerical issue.");

            const float32_t denoTerm{ kappa + static_cast<float32_t>(DIM) };

            _weight0 = kappa / denoTerm;
            _weighti = 0.5F / denoTerm;
        }

        ///
        /// @brief algorithm to calculate the deterministic sigma points for
        /// the unscented transformation
        ///
        /// @param vecX mean of the normally distributed state
        /// @param matPxx covariance of the normally distributed state
        /// @param kappa design scaling parameter for sigma points selection
        ///
        void updateSigmaPoints(const Vector<DIM> & vecX, const Matrix<DIM, DIM> & matPxx, const float32_t kappa)
        {
            const float32_t scalarMultiplier{ std::sqrt(DIM + kappa) }; // sqrt(n + \kappa)

            // cholesky factorization to get matrix Pxx square-root
            Eigen::LLT<Matrix<DIM, DIM>> lltOfPxx(matPxx);
            Matrix<DIM, DIM> matSxx{ lltOfPxx.matrixL() }; // sqrt(P_{xx})

            matSxx *= scalarMultiplier; // sqrt( (n + \kappa) * P_{xx} )

            // X_0 = \bar{x}
            util::copyToColumn< DIM, SIGMA_DIM >(0, _sigmaX, vecX);

            for (size_t i{ 0 }; i < DIM; ++i)
            {
                const size_t IDX_1{ i + 1 };
                const size_t IDX_2{ i + DIM + 1 };

                util::copyToColumn< DIM, SIGMA_DIM >(IDX_1, _sigmaX, vecX);
                util::copyToColumn< DIM, SIGMA_DIM >(IDX_2, _sigmaX, vecX);

                const Vector<DIM> vecShiftTerm{ util::getColumnAt< DIM, DIM >(i, matSxx) };

                util::addColumnFrom< DIM, SIGMA_DIM >(IDX_1, _sigmaX, vecShiftTerm);  // X_i     = \bar{x} + sqrt( (n + \kappa) * P_{xx} )
                util::subColumnFrom< DIM, SIGMA_DIM >(IDX_2, _sigmaX, vecShiftTerm);  // X_{i+n} = \bar{x} - sqrt( (n + \kappa) * P_{xx} )
            }
        }

        ///
        /// @brief transform sigma points X through the nonlinear function
        /// @param nonlinearFunction nonlinear function to be used
        /// @param sigmaY output transformed sigma points from the nonlinear function
        ///
        template<typename NonLinearFunctionCallback>
        void transformSigmaPoints(NonLinearFunctionCallback nonlinearFunction, Matrix<DIM, SIGMA_DIM> & sigmaY)
        {
            for (size_t i{ 0 }; i < SIGMA_DIM; ++i)
            {
                const Vector<DIM> x{ util::getColumnAt<DIM, SIGMA_DIM>(i, _sigmaX) };
                const Vector<DIM> y{ nonlinearFunction(x) }; // y = f(x)

                util::copyToColumn<DIM, SIGMA_DIM>(i, sigmaY, y); // Y[:, i] = y
            }
        }

        ///
        /// @brief calculate weighted mean and covariance given a set of sigma points with associated weights.
        /// @param sigmaY input transformed sigma points from the nonlinear function
        /// @param vecY output weighted mean vector
        /// @param matPyy output weighted covariance matrix
        ///
        void updateTransformedMeanAndCovariance(const Matrix<DIM, SIGMA_DIM> & sigmaY, Vector<DIM> & vecY, Matrix<DIM, DIM> & matPyy)
        {
            // 1. calculate mean: \bar{y} = \sum_{i_0}^{2n} W[0, i] Y[:, i]
            vecY = _weight0 * util::getColumnAt<DIM, SIGMA_DIM>(0, sigmaY);
            for (size_t i{ 1 }; i < SIGMA_DIM; ++i)
            {
                vecY += _weighti * util::getColumnAt<DIM, SIGMA_DIM>(i, sigmaY); // y += W[0, i] Y[:, i]
            }

            // 2. calculate covariance: P_{yy} = \sum_{i_0}^{2n} W[0, i] (Y[:, i] - \bar{y}) (Y[:, i] - \bar{y})^T
            Vector<DIM> devYi{ util::getColumnAt<DIM, SIGMA_DIM>(0, sigmaY) - vecY }; // Y[:, 0] - \bar{ y }
            matPyy = _weight0 * devYi * devYi.transpose(); // P_0 = W[0, 0] (Y[:, 0] - \bar{y}) (Y[:, 0] - \bar{y})^T

            for (size_t i{ 1 }; i < SIGMA_DIM; ++i)
            {
                devYi = util::getColumnAt<DIM, SIGMA_DIM>(i, sigmaY) - vecY; // Y[:, i] - \bar{y}

                const Matrix<DIM, DIM> Pi{ _weighti * devYi * devYi.transpose() }; // P_i = W[0, i] (Y[:, i] - \bar{y}) (Y[:, i] - \bar{y})^T

                matPyy += Pi; // y += W[0, i] (Y[:, i] - \bar{y}) (Y[:, i] - \bar{y})^T
            }
        }
    };
}

#endif // __UNSCENTED_TRANSFORM_H__
