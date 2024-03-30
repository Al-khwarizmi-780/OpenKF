#include "gtest/gtest.h"
#include "kalman_filter/unscented_transform.h"

class UnscentedTransformTest : public testing::Test
{
public:
    virtual void SetUp() override {}
    virtual void TearDown() override {}

    static constexpr float ASSERT_FLT_EPSILON{ 0.00001F };

    static constexpr size_t DIM_1{ 1 };
    static constexpr size_t DIM_2{ 2 };

    static kf::Vector<DIM_1> function1(const kf::Vector<DIM_1>& x)
    {
        kf::Vector<DIM_1> y;
        y[0] = x[0] * x[0];
        return y;
    }

    static kf::Vector<DIM_2> function2(const kf::Vector<DIM_2>& x)
    {
        kf::Vector<DIM_2> y;
        y[0] = x[0] * x[0];
        y[1] = x[1] * x[1];
        return y;
    }
};

TEST_F(UnscentedTransformTest, test_unscentedTransform_Example1)
{
    kf::Vector<DIM_1> x;
    x << 0.0F;

    kf::Matrix<DIM_1, DIM_1> P;
    P << 0.5F;

    kf::UnscentedTransform<DIM_1> UT;
    UT.compute(x, P, 0.0F);

    kf::Vector<DIM_1> vecY;
    kf::Matrix<DIM_1, DIM_1> matPyy;

    UT.transform(function1, vecY, matPyy);

    ASSERT_NEAR(vecY[0], 0.5F, ASSERT_FLT_EPSILON);
    ASSERT_NEAR(matPyy[0], 0.0F, ASSERT_FLT_EPSILON);
}

TEST_F(UnscentedTransformTest, test_unscentedTransform_Example2)
{
    kf::Vector<DIM_2> x;
    x << 2.0F, 1.0F;

    kf::Matrix<DIM_2, DIM_2> P;
    P << 0.1F, 0.0F, 0.0F, 0.1F;

    kf::UnscentedTransform<DIM_2> UT;
    UT.compute(x, P, 0.0F);

    kf::Vector<DIM_2> vecY;
    kf::Matrix<DIM_2, DIM_2> matPyy;

    UT.transform(function2, vecY, matPyy);

    ASSERT_NEAR(vecY[0], 4.1F, ASSERT_FLT_EPSILON);
    ASSERT_NEAR(vecY[1], 1.1F, ASSERT_FLT_EPSILON);
    ASSERT_NEAR(matPyy(0, 0), 1.61F, ASSERT_FLT_EPSILON);
    ASSERT_NEAR(matPyy(0, 1), -0.01F, ASSERT_FLT_EPSILON);
    ASSERT_NEAR(matPyy(1, 0), -0.01F, ASSERT_FLT_EPSILON);
    ASSERT_NEAR(matPyy(1, 1), 0.41F, ASSERT_FLT_EPSILON);
}