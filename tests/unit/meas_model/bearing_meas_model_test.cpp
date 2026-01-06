#include "meas_model/bearing_meas_model.h"
#include "motion_model/cv_motion_model.h"
#include "types.h"
#include <gtest/gtest.h>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace kf;
using namespace kf::motionmodel;
using namespace kf::measmodel;

class BearingMeasModelTest : public testing::Test
{
 protected:
  virtual void SetUp() override { m_initialState << 10.0, 20.0, 5.0, -2.0; }
  virtual void TearDown() override {}

  static constexpr int32_t DIM_X{motionmodel::DIM_X_CV};
  static constexpr int32_t DIM_Z{measmodel::DIM_Z_BEARING};

  Vector<DIM_X> m_initialState;
  Vector<2> m_sensPos2D{0.0F, 0.0F};
  measmodel::BearingMeasModel<DIM_X> m_bearingMeasModel{m_sensPos2D, 0.2F};
};

TEST_F(BearingMeasModelTest, test_MeasurementFunction)
{
  Vector<DIM_Z> const vecZMeas{m_bearingMeasModel.h(m_initialState)};

  const float32_t expectedBearing{
      std::atan2(20.0F - 0.0F, 10.0F - 0.0F)};  // atan2(py, px)

  EXPECT_DOUBLE_EQ(vecZMeas(0), expectedBearing);  // bearing
}

TEST_F(BearingMeasModelTest, test_JacobianHk)
{
  Matrix<DIM_Z, DIM_X> const matH{
      m_bearingMeasModel.getJacobianHk(m_initialState)};

  const float32_t px{10.0F - 0.0F};
  const float32_t py{20.0F - 0.0F};
  const float32_t denom{px * px + py * py};

  EXPECT_DOUBLE_EQ(matH(0, 0), -py / denom);  // dbearing/dpos_x
  EXPECT_DOUBLE_EQ(matH(0, 1), px / denom);   // dbearing/dpos_y
  EXPECT_DOUBLE_EQ(matH(0, 2), 0.0);          // dbearing/dvel_x
  EXPECT_DOUBLE_EQ(matH(0, 3), 0.0);          // dbearing/dvel_y
}

TEST_F(BearingMeasModelTest, test_MeasurementNoiseCov)
{
  Matrix<DIM_Z, DIM_Z> const matR{m_bearingMeasModel.getMeasurementNoiseCov()};

  const float32_t sigma2{0.2F * 0.2F};

  EXPECT_DOUBLE_EQ(matR(0, 0), sigma2);  // var_bearing
}
