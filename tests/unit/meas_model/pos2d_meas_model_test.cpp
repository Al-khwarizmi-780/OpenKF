#include "meas_model/pos2d_meas_model.h"
#include "motion_model/cv_motion_model.h"
#include "types.h"
#include <gtest/gtest.h>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace kf;
using namespace kf::motionmodel;
using namespace kf::measmodel;

class Pos2DMeasModelTest : public testing::Test
{
 protected:
  virtual void SetUp() override
  {
    m_pos2DMeasModel.setPosSigma(0.3F);
    m_initialState << 15.0, -10.0, 3.0, 4.0;
  }
  virtual void TearDown() override {}

  static constexpr int32_t DIM_X{motionmodel::DIM_X_CV};
  static constexpr int32_t DIM_Z{measmodel::DIM_Z_POS2D};

  Vector<DIM_X> m_initialState;
  Vector<2> const m_sensPos{1.0F, 2.0F};
  measmodel::Pos2dMeasModel<DIM_X> m_pos2DMeasModel{m_sensPos};
};

TEST_F(Pos2DMeasModelTest, test_MeasurementFunction)
{
  Vector<DIM_Z> const vecZMeas{m_pos2DMeasModel.h(m_initialState)};

  EXPECT_DOUBLE_EQ(vecZMeas(0), 14.0);   // pos_x
  EXPECT_DOUBLE_EQ(vecZMeas(1), -12.0);  // pos_y
}

TEST_F(Pos2DMeasModelTest, test_JacobianHk)
{
  Matrix<DIM_Z, DIM_X> const matH{
      m_pos2DMeasModel.getJacobianHk(m_initialState)};

  EXPECT_DOUBLE_EQ(matH(0, 0), 1.0);  // dpos_x/dpos_x
  EXPECT_DOUBLE_EQ(matH(0, 1), 0.0);  // dpos_x/dpos_y
  EXPECT_DOUBLE_EQ(matH(0, 2), 0.0);  // dpos_x/dvel_x
  EXPECT_DOUBLE_EQ(matH(0, 3), 0.0);  // dpos_x/dvel_y

  EXPECT_DOUBLE_EQ(matH(1, 0), 0.0);  // dpos_y/dpos_x
  EXPECT_DOUBLE_EQ(matH(1, 1), 1.0);  // dpos_y/dpos_y
  EXPECT_DOUBLE_EQ(matH(1, 2), 0.0);  // dpos_y/dvel_x
  EXPECT_DOUBLE_EQ(matH(1, 3), 0.0);  // dpos_y/dvel_y
}

TEST_F(Pos2DMeasModelTest, test_MeasurementNoiseCov)
{
  Matrix<DIM_Z, DIM_Z> const matR{m_pos2DMeasModel.getMeasurementNoiseCov()};

  const float32_t sigma2{0.3F * 0.3F};

  EXPECT_DOUBLE_EQ(matR(0, 0), sigma2);  // var_pos_x
  EXPECT_DOUBLE_EQ(matR(0, 1), 0.0);     // cov_pos_xy

  EXPECT_DOUBLE_EQ(matR(1, 0), 0.0);     // cov_pos_yx
  EXPECT_DOUBLE_EQ(matR(1, 1), sigma2);  // var_pos_y
}
