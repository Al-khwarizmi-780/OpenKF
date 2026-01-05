#include "kalman_filter/kalman_filter.h"
#include "motion_model/ca_motion_model.h"
#include "types.h"
#include <gtest/gtest.h>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace kf;

class CaMotionModelTest : public testing::Test
{
 protected:
  virtual void SetUp() override
  {
    m_caMotionModel.setSigma(0.5F);
    m_initState << 10.0, 20.0, 5.0, -2.0, 1.0, -0.5;
  }
  virtual void TearDown() override {}

  static constexpr int32_t DIM_X{motionmodel::DIM_X_CA};

  Vector<DIM_X> m_initState;
  motionmodel::CaMotionModel m_caMotionModel;
};

TEST_F(CaMotionModelTest, test_PredictZeroDeltaTime)
{
  Vector<DIM_X> predicted_state = m_caMotionModel.f(m_initState, 0.0);

  EXPECT_DOUBLE_EQ(predicted_state(0), m_initState(0));  // x unchanged
  EXPECT_DOUBLE_EQ(predicted_state(1), m_initState(1));  // y unchanged
  EXPECT_DOUBLE_EQ(predicted_state(2), m_initState(2));  // vx unchanged
  EXPECT_DOUBLE_EQ(predicted_state(3), m_initState(3));  // vy unchanged
  EXPECT_DOUBLE_EQ(predicted_state(4), m_initState(4));  // ax unchanged
  EXPECT_DOUBLE_EQ(predicted_state(5), m_initState(5));  // ay unchanged
}

// Basic valid delta time tests
TEST_F(CaMotionModelTest, test_PredictSmallPositiveDeltaTime)
{
  float32_t dt{0.1F};  // Small positive time step

  Vector<DIM_X> predicted_state = m_caMotionModel.f(m_initState, dt);

  // Expected values using kinematic equations:
  // x = x0 + vx*dt + 0.5*ax*dt²
  // vx = vx0 + ax*dt
  // ax remains constant

  float32_t expected_x =
      m_initState(0) + m_initState(2) * dt + 0.5F * m_initState(4) * dt * dt;
  float32_t expected_y =
      m_initState(1) + m_initState(3) * dt + 0.5F * m_initState(5) * dt * dt;
  float32_t expected_vx = m_initState(2) + m_initState(4) * dt;
  float32_t expected_vy = m_initState(3) + m_initState(5) * dt;

  EXPECT_NEAR(predicted_state(0), expected_x, 1e-5F);
  EXPECT_NEAR(predicted_state(1), expected_y, 1e-5F);
  EXPECT_NEAR(predicted_state(2), expected_vx, 1e-5F);
  EXPECT_NEAR(predicted_state(3), expected_vy, 1e-5F);
  EXPECT_NEAR(predicted_state(4), 1.0F, 1e-5F);   // ax unchanged
  EXPECT_NEAR(predicted_state(5), -0.5F, 1e-5F);  // ay unchanged
}

TEST_F(CaMotionModelTest, test_PredictMediumDeltaTime)
{
  float32_t dt{2.5};  // Medium time step

  Vector<DIM_X> predicted_state = m_caMotionModel.f(m_initState, dt);

  // Expected values using kinematic equations:
  // x = x0 + vx*dt + 0.5*ax*dt * dt
  // vx = vx0 + ax*dt
  float32_t expected_x =
      m_initState(0) + m_initState(2) * dt + 0.5F * m_initState(4) * dt * dt;
  float32_t expected_y =
      m_initState(1) + m_initState(3) * dt + 0.5F * m_initState(5) * dt * dt;
  float32_t expected_vx = m_initState(2) + m_initState(4) * dt;
  float32_t expected_vy = m_initState(3) + m_initState(5) * dt;

  EXPECT_NEAR(predicted_state(0), expected_x, 1e-9F);
  EXPECT_NEAR(predicted_state(1), expected_y, 1e-9F);
  EXPECT_NEAR(predicted_state(2), expected_vx, 1e-9F);
  EXPECT_NEAR(predicted_state(3), expected_vy, 1e-9F);
}

TEST_F(CaMotionModelTest, test_PredictLargeDeltaTime)
{
  float32_t dt{10.0};  // Large time step

  Vector<DIM_X> predicted_state = m_caMotionModel.f(m_initState, dt);

  float32_t expected_x =
      m_initState(0) + m_initState(2) * dt + 0.5F * m_initState(4) * dt * dt;
  float32_t expected_y =
      m_initState(1) + m_initState(3) * dt + 0.5F * m_initState(5) * dt * dt;
  float32_t expected_vx = m_initState(2) + m_initState(4) * dt;
  float32_t expected_vy = m_initState(3) + m_initState(5) * dt;

  EXPECT_NEAR(predicted_state(0), expected_x, 1e-9F);
  EXPECT_NEAR(predicted_state(1), expected_y, 1e-9F);
  EXPECT_NEAR(predicted_state(2), expected_vx, 1e-9F);
  EXPECT_NEAR(predicted_state(3), expected_vy, 1e-9F);
}

TEST_F(CaMotionModelTest, test_PredictWithAcceleration)
{
  float32_t dt{2.0F};
  Vector<DIM_X> predicted_state = m_caMotionModel.f(m_initState, dt);

  // Kinematic equations:
  // x = x0 + vx*dt + 0.5*ax*dt^2
  // vx = vx0 + ax*dt

  float32_t expected_x =
      m_initState(0) + m_initState(2) * dt + 0.5F * m_initState(4) * dt * dt;
  float32_t expected_y =
      m_initState(1) + m_initState(3) * dt + 0.5F * m_initState(5) * dt * dt;
  float32_t expected_vx = m_initState(2) + m_initState(4) * dt;
  float32_t expected_vy = m_initState(3) + m_initState(5) * dt;

  EXPECT_NEAR(predicted_state(0), expected_x, 1e-9F);
  EXPECT_NEAR(predicted_state(1), expected_y, 1e-9F);
  EXPECT_NEAR(predicted_state(2), expected_vx, 1e-9F);
  EXPECT_NEAR(predicted_state(3), expected_vy, 1e-9F);
}

TEST_F(CaMotionModelTest, test_GetProcessNoiseCov)
{
  float32_t const dt{0.1F};

  Matrix<DIM_X, DIM_X> const matQ{
      m_caMotionModel.getProcessNoiseCov(m_initState, dt)};

  // Q should be positive semi-definite (symmetric with non-negative
  // eigenvalues)
  EXPECT_TRUE(matQ.isApprox(matQ.transpose()));  // Symmetry check

  // Check dimensions
  EXPECT_EQ(matQ.rows(), 6);
  EXPECT_EQ(matQ.cols(), 6);

  // Q should not be zero matrix
  EXPECT_GT(matQ.norm(), 0.0);
}

TEST_F(CaMotionModelTest, test_StateTransitionMatrix)
{
  float32_t dt{1.5F};
  Matrix<DIM_X, DIM_X> F{m_caMotionModel.getJacobianFk(m_initState, dt)};

  // For CA model (6x6 matrix):
  // [1 0 dt 0  0.5*dt^2 0]
  // [0 1 0  dt 0        0.5*dt^2]
  // [0 0 1  0  dt       0]
  // [0 0 0  1  0        dt]
  // [0 0 0  0  1        0]
  // [0 0 0  0  0        1]

  float32_t half_dt2{0.5F * dt * dt};

  EXPECT_DOUBLE_EQ(F(0, 0), 1.0F);
  EXPECT_DOUBLE_EQ(F(0, 2), dt);
  EXPECT_DOUBLE_EQ(F(0, 4), half_dt2);

  EXPECT_DOUBLE_EQ(F(1, 1), 1.0F);
  EXPECT_DOUBLE_EQ(F(1, 3), dt);
  EXPECT_DOUBLE_EQ(F(1, 5), half_dt2);

  EXPECT_EQ(F.rows(), 6);
  EXPECT_EQ(F.cols(), 6);
}

TEST_F(CaMotionModelTest, test_PredictConsistency)
{
  float32_t dt{3.0F};

  // Direct prediction
  Vector<DIM_X> predicted_state{m_caMotionModel.f(m_initState, dt)};

  // Prediction using state transition matrix
  Matrix<DIM_X, DIM_X> F{m_caMotionModel.getJacobianFk(m_initState, dt)};
  Vector<DIM_X> matrix_prediction{F * m_initState};

  // Both should give same result
  EXPECT_TRUE(predicted_state.isApprox(matrix_prediction));
}