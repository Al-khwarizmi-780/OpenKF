#include "meas_model/range_bearing_meas_model.h"
#include "motion_model/cv_motion_model.h"
#include "types.h"
#include <gtest/gtest.h>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace kf;
using namespace kf::motionmodel;
using namespace kf::measmodel;

class RangeBearingMeasModelTest : public testing::Test
{
 protected:
  virtual void SetUp() override { m_initialState << 10.0, 20.0, 5.0, -2.0; }
  virtual void TearDown() override {}

  static constexpr int32_t DIM_X{motionmodel::DIM_X_CV};
  static constexpr int32_t DIM_Z{measmodel::DIM_Z_BEARING};

  Vector<DIM_X> m_initialState;
  Vector<2> const sensPos2D{0.0F, 0.0F};
  measmodel::RangeBearingMeasModel<DIM_X> m_rangeBearingMeasModel{sensPos2D,
                                                                  0.3F, 0.2F};
};

TEST_F(RangeBearingMeasModelTest, test_MeasurementFunction)
{
  Vector<DIM_Z> const vecZMeas{m_rangeBearingMeasModel.h(m_initialState)};

  const float32_t expectedRange{
      std::sqrt(10.0F * 10.0F + 20.0F * 20.0F)};  // sqrt(px^2 + py^2)
  const float32_t expectedBearing{
      std::atan2(20.0F - 0.0F, 10.0F - 0.0F)};  // atan2(py, px)

  EXPECT_DOUBLE_EQ(vecZMeas(0), expectedRange);    // range
  EXPECT_DOUBLE_EQ(vecZMeas(1), expectedBearing);  // bearing
}

TEST_F(RangeBearingMeasModelTest, test_JacobianHk)
{
  Matrix<DIM_Z, DIM_X> const matH{
      m_rangeBearingMeasModel.getJacobianHk(m_initialState)};

  const float32_t px{10.0F - 0.0F};
  const float32_t py{20.0F - 0.0F};
  const float32_t denom{px * px + py * py};

  EXPECT_DOUBLE_EQ(matH(0, 0), px / std::sqrt(denom));  // drange/dpos_x
  EXPECT_DOUBLE_EQ(matH(0, 1), py / std::sqrt(denom));  // drange/dpos_y
  EXPECT_DOUBLE_EQ(matH(0, 2), 0.0);                    // drange/dvel_x
  EXPECT_DOUBLE_EQ(matH(0, 3), 0.0);                    // drange/dvel_y

  EXPECT_DOUBLE_EQ(matH(1, 0), -py / denom);  // dbearing/dpos_x
  EXPECT_DOUBLE_EQ(matH(1, 1), px / denom);   // dbearing/dpos_y
  EXPECT_DOUBLE_EQ(matH(1, 2), 0.0);          // dbearing/dvel_x
  EXPECT_DOUBLE_EQ(matH(1, 3), 0.0);          // dbearing/dvel_y
}

TEST_F(RangeBearingMeasModelTest, test_MeasurementNoiseCov)
{
  Matrix<DIM_Z, DIM_Z> const matR{
      m_rangeBearingMeasModel.getMeasurementNoiseCov()};

  const float32_t rangeSigma2{0.3F * 0.3F};
  const float32_t bearingSigma2{0.2F * 0.2F};

  EXPECT_DOUBLE_EQ(matR(0, 0), rangeSigma2);    // var_range
  EXPECT_DOUBLE_EQ(matR(1, 1), bearingSigma2);  // var_bearing
}
