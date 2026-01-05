#include "kalman_filter/kalman_filter.h"
#include "motion_model/ct_motion_model.h"
#include "types.h"
#include <gtest/gtest.h>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace kf;

class CtMotionModelTest : public testing::Test
{
 protected:
  virtual void SetUp() override
  {
    m_ctMotionModel.setSigmaV(0.5F);
    m_ctMotionModel.setSigmaOmega(0.5F);
    m_initState << 0.0F, 0.0F, 10.0F, static_cast<float32_t>(M_PI) / 4.0F,
        static_cast<float32_t>(M_PI) / 12.0F;  // 45° heading, 15°/s turn
  }
  virtual void TearDown() override {}

  static constexpr int32_t DIM_X{motionmodel::DIM_X_CT};

  Vector<DIM_X> m_initState;
  motionmodel::CtMotionModel m_ctMotionModel;
};

TEST_F(CtMotionModelTest, test_PredictCircularMotion)
{
  float32_t dt{1.0F};
  Vector<DIM_X> predicted_state{m_ctMotionModel.f(m_initState, dt)};

  float32_t displacement{m_initState[motionmodel::CtMotionModel::IDX_V] * dt};

  Vector<DIM_X> expectedState;
  expectedState
      << m_initState[motionmodel::CtMotionModel::IDX_PX] +
             (displacement *
              cosf(m_initState[motionmodel::CtMotionModel::IDX_THETA])),
      m_initState[motionmodel::CtMotionModel::IDX_PY] +
          (displacement *
           sinf(m_initState[motionmodel::CtMotionModel::IDX_THETA])),
      m_initState[motionmodel::CtMotionModel::IDX_V],
      m_initState[motionmodel::CtMotionModel::IDX_THETA] +
          (m_initState[motionmodel::CtMotionModel::IDX_OMEGA] * dt),
      m_initState[motionmodel::CtMotionModel::IDX_OMEGA];

  EXPECT_NEAR(predicted_state(0), expectedState(0), 1e-9F);  // x
  EXPECT_NEAR(predicted_state(1), expectedState(1), 1e-9F);  // y
  EXPECT_FLOAT_EQ(predicted_state(2), expectedState(2));  // velocity unchanged
  EXPECT_NEAR(predicted_state(3), expectedState(3), 1e-9F);  // new heading
  EXPECT_FLOAT_EQ(predicted_state(4), expectedState(4));  // turn rate unchanged
}

TEST_F(CtMotionModelTest, test_ZeroTurnRateDegeneratesToCV)
{
  Vector<DIM_X> state;
  state << 0.0F, 0.0F, 10.0F, static_cast<float32_t>(M_PI) / 4.0F,
      0.0F;  // Zero turn rate

  float32_t dt{2.0F};
  Vector<DIM_X> predicted_state{m_ctMotionModel.f(state, dt)};

  // With zero turn rate, should behave like CV model
  float32_t delta_x{10.0F * cosf(static_cast<float32_t>(M_PI) / 4.0F) * dt};
  float32_t delta_y{10.0F * sinf(static_cast<float32_t>(M_PI) / 4.0F) * dt};

  EXPECT_NEAR(predicted_state(0), delta_x, 1e-9F);
  EXPECT_NEAR(predicted_state(1), delta_y, 1e-9F);
  EXPECT_FLOAT_EQ(predicted_state(3),
                  static_cast<float32_t>(M_PI) / 4.0F);  // Heading unchanged
}
