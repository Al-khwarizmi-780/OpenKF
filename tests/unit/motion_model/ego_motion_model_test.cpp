#include <gtest/gtest.h>
#include "motion_model/ego_motion_model.h"
#include "types.h"

using namespace kf;

class EgoMotionModelTest : public testing::Test
{
public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}

    static constexpr int32_t DIM_X{ 3 };
    static constexpr int32_t DIM_U{ 2 };
};

TEST_F(EgoMotionModelTest, test_egoMotionModel_Covariances)
{
    kf::Vector<DIM_X> vecX;
    vecX << 1.0F, 2.0F, 0.015F;

    kf::Vector<DIM_U> vecU;
    vecU << 0.5F, 0.005F;

    kf::motionmodel::EgoMotionModel egoMotionModel;

    float32_t const qX{ 0.2F };
    float32_t const qY{ 0.3F };
    float32_t const qTheta{ 0.003F };

    float32_t const uDeltaDist{ 0.02F };
    float32_t const uDeltaYaw{ 0.0002F };

    egoMotionModel.setNoiseX(qX);
    egoMotionModel.setNoiseY(qY);
    egoMotionModel.setNoiseTheta(qTheta);
    egoMotionModel.setNoiseDeltaDist(uDeltaDist);
    egoMotionModel.setNoiseDeltaYaw(uDeltaYaw);

    kf::Matrix<DIM_X,DIM_X> const matQk{ egoMotionModel.getProcessNoiseCov(vecX, vecU) };
    kf::Matrix<DIM_X,DIM_X> const matUk{ egoMotionModel.getInputNoiseCov(vecX, vecU) };

    kf::Matrix<DIM_X,DIM_X> const matFk{ egoMotionModel.getJacobianFk(vecX, vecU) };
    kf::Matrix<DIM_X,DIM_U> const matBk{ egoMotionModel.getJacobianBk(vecX, vecU) };

    kf::Matrix<DIM_X,DIM_X> matQ;
    matQ << qX, 0.0F, 0.0F,
            0.0F, qY, 0.0F,
            0.0F, 0.0F, qTheta;

    kf::Matrix<DIM_U,DIM_U> matU;
    matU << uDeltaDist, 0.0F,
            0.0F, uDeltaYaw;

    kf::Matrix<DIM_X,DIM_X> const matQkExp{ matFk * matQ * matFk.transpose() };
    kf::Matrix<DIM_X,DIM_X> const matUkExp{ matBk * matU * matBk.transpose() };

    EXPECT_NEAR(matQk(0,0), matQkExp(0,0), 0.0001F);
    EXPECT_NEAR(matQk(0,1), matQkExp(0,1), 0.0001F);
    EXPECT_NEAR(matQk(0,2), matQkExp(0,2), 0.0001F);
    EXPECT_NEAR(matQk(1,0), matQkExp(1,0), 0.0001F);
    EXPECT_NEAR(matQk(1,1), matQkExp(1,1), 0.0001F);
    EXPECT_NEAR(matQk(1,2), matQkExp(1,2), 0.0001F);
    EXPECT_NEAR(matQk(2,0), matQkExp(2,0), 0.0001F);
    EXPECT_NEAR(matQk(2,1), matQkExp(2,1), 0.0001F);
    EXPECT_NEAR(matQk(2,2), matQkExp(2,2), 0.0001F);

    EXPECT_NEAR(matUk(0,0), matUkExp(0,0), 0.0001F);
    EXPECT_NEAR(matUk(0,1), matUkExp(0,1), 0.0001F);
    EXPECT_NEAR(matUk(0,2), matUkExp(0,2), 0.0001F);
    EXPECT_NEAR(matUk(1,0), matUkExp(1,0), 0.0001F);
    EXPECT_NEAR(matUk(1,1), matUkExp(1,1), 0.0001F);
    EXPECT_NEAR(matUk(1,2), matUkExp(1,2), 0.0001F);
    EXPECT_NEAR(matUk(2,0), matUkExp(2,0), 0.0001F);
    EXPECT_NEAR(matUk(2,1), matUkExp(2,1), 0.0001F);
    EXPECT_NEAR(matUk(2,2), matUkExp(2,2), 0.0001F);
}

TEST_F(EgoMotionModelTest, test_egoMotionModel_ForwardReverseMoves)
{
    kf::motionmodel::EgoMotionModel egoMotionModel;
    kf::Vector<DIM_X> vecX;
    kf::Vector<DIM_U> vecU;
    kf::Vector<DIM_X> vecXn;

    // moving forward
    vecX << 0.0F, 0.0F, 0.0F;
    vecU << 0.5F, 0.0F;

    vecXn = egoMotionModel.f(vecX, vecU);

    EXPECT_NEAR(vecXn[0], vecX[0]+0.5F, 0.0001F);
    EXPECT_NEAR(vecXn[1], vecX[1], 0.0001F);
    EXPECT_NEAR(vecXn[2], vecX[2], 0.0001F);

    // moving backward/reverse
    vecX << 0.0F, 0.0F, 0.0F;
    vecU << -0.5F, 0.0F;

    vecXn = egoMotionModel.f(vecX, vecU);

    EXPECT_NEAR(vecXn[0], vecX[0]-0.5F, 0.0001F);
    EXPECT_NEAR(vecXn[1], vecX[1], 0.0001F);
    EXPECT_NEAR(vecXn[2], vecX[2], 0.0001F);

    // moving forward + oriented 90 degrees
    vecX << 0.0F, 0.0F, M_PI / 2.0F;
    vecU << 0.5F, 0.0F;

    vecXn = egoMotionModel.f(vecX, vecU);

    EXPECT_NEAR(vecXn[0], vecX[0], 0.0001F);
    EXPECT_NEAR(vecXn[1], vecX[1] + 0.5F, 0.0001F);
    EXPECT_NEAR(vecXn[2], vecX[2], 0.0001F);

    // moving backward + oriented 90 degrees
    vecX << 0.0F, 0.0F, M_PI / 2.0F;
    vecU << -0.5F, 0.0F;

    vecXn = egoMotionModel.f(vecX, vecU);

    EXPECT_NEAR(vecXn[0], vecX[0], 0.0001F);
    EXPECT_NEAR(vecXn[1], vecX[1] - 0.5F, 0.0001F);
    EXPECT_NEAR(vecXn[2], vecX[2], 0.0001F);

    // moving forward + oriented -90 degrees
    vecX << 0.0F, 0.0F, -M_PI / 2.0F;
    vecU << 0.5F, 0.0F;

    vecXn = egoMotionModel.f(vecX, vecU);

    EXPECT_NEAR(vecXn[0], vecX[0], 0.0001F);
    EXPECT_NEAR(vecXn[1], vecX[1] - 0.5F, 0.0001F);
    EXPECT_NEAR(vecXn[2], vecX[2], 0.0001F);

    // moving backward + oriented -90 degrees
    vecX << 0.0F, 0.0F, -M_PI / 2.0F;
    vecU << -0.5F, 0.0F;

    vecXn = egoMotionModel.f(vecX, vecU);

    EXPECT_NEAR(vecXn[0], vecX[0], 0.0001F);
    EXPECT_NEAR(vecXn[1], vecX[1] + 0.5F, 0.0001F);
    EXPECT_NEAR(vecXn[2], vecX[2], 0.0001F);

    // moving forward + oriented 45 degrees
    vecX << 0.0F, 0.0F, M_PI / 4.0F;
    vecU << 0.5F, 0.0F;

    vecXn = egoMotionModel.f(vecX, vecU);

    EXPECT_NEAR(vecXn[0], vecX[0] + vecU[0] * std::cos(vecX[2] + (vecU[1] / 2.0F)), 0.0001F);
    EXPECT_NEAR(vecXn[1], vecX[1] + vecU[0] * std::sin(vecX[2] + (vecU[1] / 2.0F)), 0.0001F);
    EXPECT_NEAR(vecXn[2], vecX[2] + vecU[1], 0.0001F);

    // moving forward + oriented 45 degrees
    vecX << 0.0F, 0.0F, -M_PI / 4.0F;
    vecU << 0.5F, 0.0F;

    vecXn = egoMotionModel.f(vecX, vecU);

    EXPECT_NEAR(vecXn[0], vecX[0] + vecU[0] * std::cos(vecX[2] + (vecU[1] / 2.0F)), 0.0001F);
    EXPECT_NEAR(vecXn[1], vecX[1] + vecU[0] * std::sin(vecX[2] + (vecU[1] / 2.0F)), 0.0001F);
    EXPECT_NEAR(vecXn[2], vecX[2] + vecU[1], 0.0001F);
}