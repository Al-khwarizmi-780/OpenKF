///
/// Copyright 2022 CodingCorner
///
/// Use of this source code is governed by an MIT - style
/// license that can be found in the LICENSE file or at
/// https ://opensource.org/licenses/MIT.
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

        UnscentedTransform() = delete;

        template<typename NonLinearFunctionCallback>
        explicit UnscentedTransform(
            NonLinearFunctionCallback nonlinearFunction,const Vector<DIM> & vecX, const Matrix<DIM, DIM> & matPxx, const float32_t kappa=0.0F)
        {
            // 1. calculate weights
            updateWeights(kappa);

            // 2. update sigma points _sigmaX
            updateSigmaPoints(vecX, matPxx, kappa);

            // 3. transform sigma points X through the nonlinear model f(X) to obtain Y sigma points
            transformSigmaPoints(nonlinearFunction);

            // 4. calculate weight mean and covariance from the transformed sigma points Y and weights
            updateTransformedMeanAndCovariance();
        }

        ~UnscentedTransform() {}

        void showSummary()
        {
            std::cout << "DIM_N : " << DIM << "\n";
            std::cout << "DIM_M : " << SIGMA_DIM << "\n";
            std::cout << "_weights : \n" << _weights << "\n";
            std::cout << "_sigmaX : \n" << _sigmaX << "\n";
            std::cout << "_sigmaY : \n" << _sigmaY << "\n";
            std::cout << "_vecY : \n" << _vecY << "\n";
            std::cout << "_matPyy : \n" << _matPyy << "\n";
        }

        Matrix<1, SIGMA_DIM> & weights() { return _weights; }
        const Matrix<1, SIGMA_DIM> & weights() const { return _weights; }

        Matrix<DIM, SIGMA_DIM> & sigmaX() { return _sigmaX; }
        const Matrix<DIM, SIGMA_DIM> & sigmaX() const { return _sigmaX; }

        Matrix<DIM, SIGMA_DIM> & sigmaY() { return _sigmaY; }
        const Matrix<DIM, SIGMA_DIM> & sigmaY() const { return _sigmaY; }

        Vector<DIM> & vecY() { return _vecY; }
        const Vector<DIM> & vecY() const { return _vecY; }

        Matrix<DIM, DIM> & matPyy() { return _matPyy; }
        const Matrix<DIM, DIM> & matPyy() const { return _matPyy; }

    private:
        Matrix<1, SIGMA_DIM> _weights;  /// @brief unscented transform weights

        Matrix<DIM, SIGMA_DIM> _sigmaX; /// @brief input sigma points
        Matrix<DIM, SIGMA_DIM> _sigmaY; /// @brief output sigma points

        Vector<DIM> _vecY;              /// @brief output y mean vector (weighted mean vector)
        Matrix<DIM, DIM> _matPyy;       /// @brief output Pyy covariance matrix (weighted covariance matrix)

        ///
        /// @brief algorithm to calculate the weights used to draw the sigma points
        ///
        /// @param kappa design scaling parameter for sigma points selection
        ///
        void updateWeights(float32_t kappa)
        {
            const float32_t denoTerm{ kappa + static_cast<float32_t>(DIM) };

            _weights(0, 0) = kappa / denoTerm;

            for (int32_t i{ 1 }; i < SIGMA_DIM; ++i)
            {
                _weights(0, i) = 0.5F / denoTerm;
            }
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
        ///
        template<typename NonLinearFunctionCallback>
        void transformSigmaPoints(NonLinearFunctionCallback nonlinearFunction)
        {
            for (size_t i{ 0 }; i < SIGMA_DIM; ++i)
            {
                const Vector<DIM> x{ util::getColumnAt<DIM, SIGMA_DIM>(i, _sigmaX) };
                const Vector<DIM> y{ nonlinearFunction(x) }; // y = f(x)

                util::copyToColumn<DIM, SIGMA_DIM>(i, _sigmaY, y); // Y[:, i] = y
            }
        }

        void updateTransformedMeanAndCovariance()
        {
            // 1. calculate mean: \bar{y} = \sum_{i_0}^{2n} W[0, i] Y[:, i]
            _vecY = Vector<DIM>::Zero();
            for (size_t i{ 0 }; i < SIGMA_DIM; ++i)
            {
                _vecY += _weights(0, i) * util::getColumnAt<DIM, SIGMA_DIM>(i, _sigmaY); // y += W[0, i] Y[:, i]
            }

            // 2. calculate covariance: P_{yy} = \sum_{i_0}^{2n} W[0, i] (Y[:, i] - \bar{y}) (Y[:, i] - \bar{y})^T
            _matPyy = Matrix<DIM, DIM>::Zero();
            for (size_t i{ 0 }; i < SIGMA_DIM; ++i)
            {
                const Vector<DIM> devYi{ util::getColumnAt<DIM, SIGMA_DIM>(i, _sigmaY) - _vecY }; // Y[:, i] - \bar{y}

                Matrix<DIM, DIM> Pi{ _weights(0, i) * devYi * devYi.transpose() }; // P_i = W[0, i] (Y[:, i] - \bar{y}) (Y[:, i] - \bar{y})^T

                _matPyy += Pi; // y += W[0, i] (Y[:, i] - \bar{y}) (Y[:, i] - \bar{y})^T
            }
        }
    };
}

#endif // __UNSCENTED_TRANSFORM_H__
