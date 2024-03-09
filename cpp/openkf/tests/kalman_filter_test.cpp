#include <gtest/gtest.h>
#include "kalman_filter/kalman_filter.h"

class KalmanFilterTest : public testing::Test
{
public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}

    static constexpr float FLOAT_EPSILON{ 0.00001F };

    static constexpr int32_t DIM_X_LKF{ 2 };
    static constexpr int32_t DIM_Z_LKF{ 1 };
    static constexpr int32_t DIM_X_EKF{ 2 };
    static constexpr int32_t DIM_Z_EKF{ 2 };
    static constexpr kf::float32_t T{ 1.0F };
    static constexpr kf::float32_t Q11{ 0.1F }, Q22{ 0.1F };

    kf::KalmanFilter<DIM_X_LKF, DIM_Z_LKF> m_lkf;
    kf::KalmanFilter<DIM_X_EKF, DIM_Z_EKF> m_ekf;

    static kf::Vector<DIM_Z_EKF> covertCartesian2Polar(const kf::Vector<DIM_X_EKF>& cartesian)
    {
        kf::Vector<DIM_Z_EKF> polar;
        polar[0] = std::sqrt(cartesian[0] * cartesian[0] + cartesian[1] * cartesian[1]);
        polar[1] = std::atan2(cartesian[1], cartesian[0]);
        return polar;
    }

    static kf::Matrix<DIM_Z_EKF, DIM_Z_EKF> calculateJacobianMatrix(const kf::Vector<DIM_X_EKF>& vecX)
    {
        const kf::float32_t valX2PlusY2{ (vecX[0] * vecX[0]) + (vecX[1] * vecX[1]) };
        const kf::float32_t valSqrtX2PlusY2{ std::sqrt(valX2PlusY2) };

        kf::Matrix<DIM_Z_EKF, DIM_Z_EKF> matHj;
        matHj <<
            (vecX[0] / valSqrtX2PlusY2), (vecX[1] / valSqrtX2PlusY2),
            (-vecX[1] / valX2PlusY2), (vecX[0] / valX2PlusY2);

        return matHj;
    }
};
TEST_F(KalmanFilterTest, test_predictLKF)
{
    m_lkf.vecX() << 0.0F, 2.0F;
    m_lkf.matP() << 0.1F, 0.0F, 0.0F, 0.1F;

    kf::Matrix<DIM_X_LKF, DIM_X_LKF> F; // state transition matrix
    F << 1.0F, T, 0.0F, 1.0F;

    kf::Matrix<DIM_X_LKF, DIM_X_LKF> Q; // process noise covariance
    Q(0, 0) = (Q11 * T) + (Q22 * (std::pow(T, 3.0F) / 3.0F));
    Q(0, 1) = Q(1, 0) = Q22 * (std::pow(T, 2.0F) / 2.0F);
    Q(1, 1) = Q22 * T;

    m_lkf.predictLKF(F, Q); // execute prediction step

    ASSERT_NEAR(m_lkf.vecX()(0), 2.0F, FLOAT_EPSILON);
    ASSERT_NEAR(m_lkf.vecX()(1), 2.0F, FLOAT_EPSILON);

    ASSERT_NEAR(m_lkf.matP()(0), 0.33333F, FLOAT_EPSILON);
    ASSERT_NEAR(m_lkf.matP()(1), 0.15F, FLOAT_EPSILON);
    ASSERT_NEAR(m_lkf.matP()(2), 0.15F, FLOAT_EPSILON);
    ASSERT_NEAR(m_lkf.matP()(3), 0.2F, FLOAT_EPSILON);
}

TEST_F(KalmanFilterTest, test_correctLKF)
{
    m_lkf.vecX() << 2.0F, 2.0F;
    m_lkf.matP() << 0.33333F, 0.15F, 0.15F, 0.2F;

    kf::Vector<DIM_Z_LKF> vecZ;
    vecZ << 2.25F;

    kf::Matrix<DIM_Z_LKF, DIM_Z_LKF> matR;
    matR << 0.01F;

    kf::Matrix<DIM_Z_LKF, DIM_X_LKF> matH;
    matH << 1.0F, 0.0F;

    m_lkf.correctLKF(vecZ, matR, matH);

    ASSERT_NEAR(m_lkf.vecX()(0), 2.24272F, FLOAT_EPSILON);
    ASSERT_NEAR(m_lkf.vecX()(1), 2.10922F, FLOAT_EPSILON);

    ASSERT_NEAR(m_lkf.matP()(0), 0.00970874F, FLOAT_EPSILON);
    ASSERT_NEAR(m_lkf.matP()(1), 0.00436893F, FLOAT_EPSILON);
    ASSERT_NEAR(m_lkf.matP()(2), 0.00436893F, FLOAT_EPSILON);
    ASSERT_NEAR(m_lkf.matP()(3), 0.134466F, FLOAT_EPSILON);
}

TEST_F(KalmanFilterTest, test_correctEKF)
{
    m_ekf.vecX() << 10.0F, 5.0F;
    m_ekf.matP() << 0.3F, 0.0F, 0.0F, 0.3F;

    const kf::Vector<DIM_X_EKF> measPosCart{ 10.4F, 5.2F };
    const kf::Vector<DIM_Z_EKF> vecZ{ covertCartesian2Polar(measPosCart) };

    kf::Matrix<DIM_Z_EKF, DIM_Z_EKF> matR;
    matR << 0.1F, 0.0F, 0.0F, 0.0008F;

    kf::Matrix<DIM_Z_EKF, DIM_X_EKF> matHj{ calculateJacobianMatrix(m_ekf.vecX()) }; // jacobian matrix Hj

    m_ekf.correctEkf(covertCartesian2Polar, vecZ, matR, matHj);

    ASSERT_NEAR(m_ekf.vecX()(0), 10.3F, FLOAT_EPSILON);
    ASSERT_NEAR(m_ekf.vecX()(1), 5.15F, FLOAT_EPSILON);

    ASSERT_NEAR(m_ekf.matP()(0), 0.075F, FLOAT_EPSILON);
    ASSERT_NEAR(m_ekf.matP()(1), -1.78814e-08F, FLOAT_EPSILON);
    ASSERT_NEAR(m_ekf.matP()(2), -1.78814e-08F, FLOAT_EPSILON);
    ASSERT_NEAR(m_ekf.matP()(3), 0.075F, FLOAT_EPSILON);
}
