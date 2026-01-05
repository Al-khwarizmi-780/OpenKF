#include "kalman_filter/kalman_filter.h"
#include "motion_model/ca_motion_model.h"
#include "motion_model/ct_motion_model.h"
#include "motion_model/cv_motion_model.h"
#include "types.h"
#include <gtest/gtest.h>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace kf;

TEST(MotionModelFactoryTest, CreateAndTestAllModels)
{
  // Test that all models can be created and have basic properties
  auto cv = std::make_unique<motionmodel::CvMotionModel>();
  auto ca = std::make_unique<motionmodel::CaMotionModel>();
  auto ctr = std::make_unique<motionmodel::CtMotionModel>();

  Vector<motionmodel::DIM_X_CV> m_cvState;
  Vector<motionmodel::DIM_X_CA> m_caState;
  Vector<motionmodel::DIM_X_CT> m_ctState;

  m_cvState << 10.0F, 20.0F, 5.0F, -2.0F;
  m_caState << 10.0F, 20.0F, 5.0F, -2.0F, 1.0F, -0.5F;
  m_ctState << 0.0F, 0.0F, 10.0F, static_cast<float32_t>(M_PI) / 4.0F,
      static_cast<float32_t>(M_PI) / 12.0F;

  EXPECT_EQ(cv->getStateDim(), 4);
  EXPECT_EQ(ca->getStateDim(), 6);
  EXPECT_EQ(ctr->getStateDim(), 5);

  // Test that prediction doesn't crash for small dt
  EXPECT_NO_THROW(cv->f(m_cvState, 0.1F));
  EXPECT_NO_THROW(ca->f(m_caState, 0.1F));
  EXPECT_NO_THROW(ctr->f(m_ctState, 0.1F));
}

class MotionModelParameterizedTest
    : public ::testing::TestWithParam<std::tuple<float32_t, float32_t>>
{
};

TEST_P(MotionModelParameterizedTest, MultipleTimeSteps)
{
  auto p = GetParam();
  float32_t position = std::get<0>(p);
  float32_t velocity = std::get<1>(p);

  motionmodel::CvMotionModel cvModel;
  Vector<4> state;
  state << position, position, velocity, velocity;

  std::vector<float32_t> time_steps = {0.0F, 0.1F, 0.5F, 1.0F, 2.0F, 5.0F};

  for (float32_t dt : time_steps)
  {
    auto predicted = cvModel.f(state, dt);
    float32_t expected = position + velocity * dt;

    EXPECT_NEAR(predicted(0), expected, 1e-9F);  // x
    EXPECT_NEAR(predicted(1), expected, 1e-9F);  // y
  }
}

INSTANTIATE_TEST_SUITE_P(
    MotionModelTests, MotionModelParameterizedTest,
    ::testing::Combine(
        ::testing::Values(0.0F, 10.0F, -5.0F, 100.0F),  // position values
        ::testing::Values(0.0F, 1.0F, -2.0F, 5.0F)      // velocity values
        ));
