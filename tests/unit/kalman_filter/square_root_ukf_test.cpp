#include "kalman_filter/square_root_ukf.h"
#include "gtest/gtest.h"

class SquareRootUnscentedKalmanFilterTest : public testing::Test
{
 public:
  virtual void SetUp() override {}
  virtual void TearDown() override {}

  static constexpr float FLOAT_EPSILON{0.001F};

  static constexpr size_t DIM_X{2};
  static constexpr size_t DIM_Z{2};

  kf::SquareRootUKF<DIM_X, DIM_Z> m_srUkf;

  static kf::Vector<DIM_X> funcF(const kf::Vector<DIM_X>& x) { return x; }
};

TEST_F(SquareRootUnscentedKalmanFilterTest, test_SRUKF_Example01)
{
  // initializations
  // x0 = np.array([1.0, 2.0])
  // P0 = np.array([[1.0, 0.5], [0.5, 1.0]])
  // Q = np.array([[0.5, 0.0], [0.0, 0.5]])

  // z = np.array([1.2, 1.8])
  // R = np.array([[0.3, 0.0], [0.0, 0.3]])

  kf::Vector<DIM_X> x;
  x << 1.0F, 2.0F;

  kf::Matrix<DIM_X, DIM_X> P;
  P << 1.0F, 0.5F, 0.5F, 1.0F;

  kf::Matrix<DIM_X, DIM_X> Q;
  Q << 0.5F, 0.0F, 0.0F, 0.5F;

  kf::Vector<DIM_Z> z;
  z << 1.2F, 1.8F;

  kf::Matrix<DIM_Z, DIM_Z> R;
  R << 0.3F, 0.0F, 0.0F, 0.3F;

  m_srUkf.initialize(x, P, Q, R);

  m_srUkf.predictSRUKF(funcF);

  // Expectation from the python results:
  // =====================================
  // x1 =
  //    [1. 2.]
  // P1 =
  //    [[1.5 0.5]
  //     [0.5 1.5]]

  ASSERT_NEAR(m_srUkf.vecX()[0], 1.0F, FLOAT_EPSILON);
  ASSERT_NEAR(m_srUkf.vecX()[1], 2.0F, FLOAT_EPSILON);

  ASSERT_NEAR(m_srUkf.matP()(0, 0), 1.5F, FLOAT_EPSILON);
  ASSERT_NEAR(m_srUkf.matP()(0, 1), 0.5F, FLOAT_EPSILON);

  ASSERT_NEAR(m_srUkf.matP()(1, 0), 0.5F, FLOAT_EPSILON);
  ASSERT_NEAR(m_srUkf.matP()(1, 1), 1.5F, FLOAT_EPSILON);

  m_srUkf.correctSRUKF(funcF, z);

  // Expectations from the python results:
  // ======================================
  // x =
  //     [1.15385 1.84615]
  // P =
  //     [[ 0.24582 0.01505 ]
  //      [ 0.01505 0.24582 ]]

  ASSERT_NEAR(m_srUkf.vecX()[0], 1.15385F, FLOAT_EPSILON);
  ASSERT_NEAR(m_srUkf.vecX()[1], 1.84615F, FLOAT_EPSILON);

  ASSERT_NEAR(m_srUkf.matP()(0, 0), 0.24582F, FLOAT_EPSILON);
  ASSERT_NEAR(m_srUkf.matP()(0, 1), 0.01505F, FLOAT_EPSILON);

  ASSERT_NEAR(m_srUkf.matP()(1, 0), 0.01505F, FLOAT_EPSILON);
  ASSERT_NEAR(m_srUkf.matP()(1, 1), 0.24582F, FLOAT_EPSILON);
}