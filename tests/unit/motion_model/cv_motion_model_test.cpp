#include "kalman_filter/kalman_filter.h"
#include "motion_model/cv_motion_model.h"
#include "types.h"
#include <gtest/gtest.h>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace kf;

class CvMotionModelTest : public testing::Test
{
 protected:
  virtual void SetUp() override
  {
    m_cvMotionModel.setProcessNoiseVector(
        Vector<motionmodel::DIM_Q_CV>::Constant(0.5F));
  }
  virtual void TearDown() override {}

  static constexpr int32_t DIM_X{4};
  static constexpr int32_t DIM_Z{2};

  motionmodel::CvMotionModel m_cvMotionModel;
};

TEST_F(CvMotionModelTest, test_PredictZeroDeltaTime)
{
  Vector<DIM_X> initial_state;
  initial_state << 10.0, 20.0, 5.0, -2.0;

  Vector<DIM_X> predicted_state = m_cvMotionModel.f(initial_state, 0.0);

  EXPECT_DOUBLE_EQ(predicted_state(0), 10.0);  // x unchanged
  EXPECT_DOUBLE_EQ(predicted_state(1), 20.0);  // y unchanged
  EXPECT_DOUBLE_EQ(predicted_state(2), 5.0);   // vx unchanged
  EXPECT_DOUBLE_EQ(predicted_state(3), -2.0);  // vy unchanged
}

TEST_F(CvMotionModelTest, test_PredictValidDeltaTime)
{
  Vector<DIM_X> vecX;
  vecX << 1.0F, 2.0F, 0.5F, -0.5F;

  float32_t const dt{0.1F};

  Vector<DIM_X> const vecXPred{m_cvMotionModel.f(vecX, dt)};

  EXPECT_FLOAT_EQ(vecXPred[0], 1.0F + 0.5F * dt);
  EXPECT_FLOAT_EQ(vecXPred[1], 2.0F - 0.5F * dt);
  EXPECT_FLOAT_EQ(vecXPred[2], 0.5F);
  EXPECT_FLOAT_EQ(vecXPred[3], -0.5F);
}

TEST_F(CvMotionModelTest, test_GetProcessNoiseCov)
{
  Vector<DIM_X> vecX;
  vecX << 1.0F, 2.0F, 0.5F, -0.5F;

  float32_t const dt{0.1F};

  Matrix<DIM_X, DIM_X> const matQ{m_cvMotionModel.getProcessNoiseCov(vecX, dt)};

  const float32_t sigma2{0.5F * 0.5F};
  const float32_t dt2{dt * dt};
  const float32_t dt3{dt2 * dt};
  const float32_t dt4{dt2 * dt2};

  EXPECT_FLOAT_EQ(matQ(0, 0), sigma2 * (dt4) / 4.0F);
  EXPECT_FLOAT_EQ(matQ(0, 2), sigma2 * (dt3) / 2.0F);
  EXPECT_FLOAT_EQ(matQ(1, 1), sigma2 * (dt4) / 4.0F);
  EXPECT_FLOAT_EQ(matQ(1, 3), sigma2 * (dt3) / 2.0F);
  EXPECT_FLOAT_EQ(matQ(2, 0), sigma2 * (dt3) / 2.0F);
  EXPECT_FLOAT_EQ(matQ(2, 2), sigma2 * (dt2));
  EXPECT_FLOAT_EQ(matQ(3, 1), sigma2 * (dt3) / 2.0F);
  EXPECT_FLOAT_EQ(matQ(3, 3), sigma2 * (dt2));

  // Q should be positive semi-definite (symmetric with non-negative
  // eigenvalues)
  EXPECT_TRUE(matQ.isApprox(matQ.transpose()));  // Symmetry check

  // Check dimensions
  EXPECT_EQ(matQ.rows(), 4);
  EXPECT_EQ(matQ.cols(), 4);

  // Q should not be zero matrix
  EXPECT_GT(matQ.norm(), 0.0);
}

TEST_F(CvMotionModelTest, test_StateTransitionMatrix)
{
  // Initialize with known state
  Vector<DIM_X> init_state;
  init_state << 10.0, 20.0, 5.0, -2.0;  // x, y, vx, vy

  float32_t dt{1.5F};
  Matrix<DIM_X, DIM_X> F{m_cvMotionModel.getJacobianFk(init_state, dt)};

  // Expected F matrix for CV model:
  // [1 0 dt 0]
  // [0 1 0 dt]
  // [0 0 1 0]
  // [0 0 0 1]

  EXPECT_DOUBLE_EQ(F(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(F(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(F(0, 2), dt);
  EXPECT_DOUBLE_EQ(F(0, 3), 0.0);

  EXPECT_DOUBLE_EQ(F(1, 0), 0.0);
  EXPECT_DOUBLE_EQ(F(1, 1), 1.0);
  EXPECT_DOUBLE_EQ(F(1, 2), 0.0);
  EXPECT_DOUBLE_EQ(F(1, 3), dt);

  // Check that it's a 4x4 matrix
  EXPECT_EQ(F.rows(), 4);
  EXPECT_EQ(F.cols(), 4);
}

TEST_F(CvMotionModelTest, test_PredictConsistency)
{
  // Initialize with known state
  Vector<DIM_X> init_state;
  init_state << 10.0, 20.0, 5.0, -2.0;  // x, y, vx, vy

  float32_t dt{3.0F};

  // Direct prediction
  Vector<DIM_X> predicted_state{m_cvMotionModel.f(init_state, dt)};

  // Prediction using state transition matrix
  Matrix<DIM_X, DIM_X> F{m_cvMotionModel.getJacobianFk(init_state, dt)};
  Vector<DIM_X> matrix_prediction{F * init_state};

  // Both should give same result
  EXPECT_TRUE(predicted_state.isApprox(matrix_prediction));
}