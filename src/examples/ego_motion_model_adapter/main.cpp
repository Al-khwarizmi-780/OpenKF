///
/// Copyright 2022 Mohanad Youssef (Al-khwarizmi)
///
/// Use of this source code is governed by an GPL-3.0 - style
/// license that can be found in the LICENSE file or at
/// https://opensource.org/licenses/GPL-3.0
///
/// @author Mohanad Youssef <mohanad.magdy.hammad@gmail.com>
/// @file main.cpp
///

#include <iostream>
#include <stdint.h>

#include "types.h"
#include "motion_model/ego_motion_model.h"

static constexpr size_t DIM_X{ 5 };
static constexpr size_t DIM_U{ 3 };

using namespace kf;


/// @brief This is an adapter example to show case how to convert from the 3-dimension state egomotion model
/// to a higher or lower dimension state vector (e.g. 5-dimension state vector and 3-dimension input vector).
class EgoMotionModelAdapter : public motionmodel::MotionModelExtInput<DIM_X, DIM_U>
{
public:
    virtual Vector<DIM_X> f(Vector<DIM_X> const & vecX, Vector<DIM_U> const & vecU, Vector<DIM_X> const & /*vecQ = Vector<DIM_X>::Zero()*/) const override
    {
        Vector<3> tmpVecX;
        tmpVecX << vecX[0], vecX[1], vecX[4];

        Vector<2> tmpVecU;
        tmpVecU << vecU[1], vecU[2];

        motionmodel::EgoMotionModel const egoMotionModel;
        tmpVecX = egoMotionModel.f(tmpVecX, tmpVecU);

        Vector<DIM_X> vecXout;
        vecXout[0] = tmpVecX[0];
        vecXout[1] = tmpVecX[1];
        vecXout[4] = tmpVecX[2];

        return vecXout;
    }

    virtual Matrix<DIM_X, DIM_X> getProcessNoiseCov(Vector<DIM_X> const & vecX, Vector<DIM_U> const & vecU) const override
    {
        Vector<3> tmpVecX;
        tmpVecX << vecX[0], vecX[1], vecX[4];

        Vector<2> tmpVecU;
        tmpVecU << vecU[1], vecU[2];

        motionmodel::EgoMotionModel const egoMotionModel;

        Matrix<3, 3> matQ = egoMotionModel.getProcessNoiseCov(tmpVecX, tmpVecU);

        Matrix<DIM_X, DIM_X> matQout;
        matQout(0, 0) = matQ(0, 0);
        matQout(0, 1) = matQ(0, 1);
        matQout(0, 4) = matQ(0, 2);
        matQout(1, 0) = matQ(1, 0);
        matQout(1, 1) = matQ(1, 1);
        matQout(1, 4) = matQ(1, 2);
        matQout(4, 0) = matQ(2, 0);
        matQout(4, 1) = matQ(2, 1);
        matQout(4, 4) = matQ(2, 1);

        return matQout;
    }

    virtual Matrix<DIM_X, DIM_X> getInputNoiseCov(Vector<DIM_X> const & vecX, Vector<DIM_U> const & vecU) const override
    {
        Vector<3> tmpVecX;
        tmpVecX << vecX[0], vecX[1], vecX[4];

        Vector<2> tmpVecU;
        tmpVecU << vecU[1], vecU[2];

        motionmodel::EgoMotionModel const egoMotionModel;

        Matrix<3, 3> matU = egoMotionModel.getInputNoiseCov(tmpVecX, tmpVecU);

        Matrix<DIM_X, DIM_X> matUout;
        matUout(0, 0) = matU(0, 0);
        matUout(0, 1) = matU(0, 1);
        matUout(0, 4) = matU(0, 2);
        matUout(1, 0) = matU(1, 0);
        matUout(1, 1) = matU(1, 1);
        matUout(1, 4) = matU(1, 2);
        matUout(4, 0) = matU(2, 0);
        matUout(4, 1) = matU(2, 1);
        matUout(4, 4) = matU(2, 1);

        return matUout;
    }

    virtual Matrix<DIM_X, DIM_X> getJacobianFk(Vector<DIM_X> const & vecX, Vector<DIM_U> const & vecU) const override
    {
        Vector<3> tmpVecX;
        tmpVecX << vecX[0], vecX[1], vecX[4];

        Vector<2> tmpVecU;
        tmpVecU << vecU[1], vecU[2];

        motionmodel::EgoMotionModel const egoMotionModel;

        Matrix<3, 3> matFk = egoMotionModel.getJacobianFk(tmpVecX, tmpVecU);

        Matrix<DIM_X, DIM_X> matFkout;
        matFkout(0, 0) = matFk(0, 0);
        matFkout(0, 1) = matFk(0, 1);
        matFkout(0, 4) = matFk(0, 2);
        matFkout(1, 0) = matFk(1, 0);
        matFkout(1, 1) = matFk(1, 1);
        matFkout(1, 4) = matFk(1, 2);
        matFkout(4, 0) = matFk(2, 0);
        matFkout(4, 1) = matFk(2, 1);
        matFkout(4, 4) = matFk(2, 1);

        return matFkout;
    }

    virtual Matrix<DIM_X, DIM_U> getJacobianBk(Vector<DIM_X> const & vecX, Vector<DIM_U> const & vecU) const override
    {
        Vector<3> tmpVecX;
        tmpVecX << vecX[0], vecX[1], vecX[4];

        Vector<2> tmpVecU;
        tmpVecU << vecU[1], vecU[2];

        motionmodel::EgoMotionModel const egoMotionModel;

        Matrix<3, 2> matBk = egoMotionModel.getJacobianBk(tmpVecX, tmpVecU);

        Matrix<DIM_X, DIM_U> matBkout;
        matBkout(0, 1) = matBk(0, 0);
        matBkout(0, 2) = matBk(0, 1);
        matBkout(1, 1) = matBk(1, 0);
        matBkout(1, 2) = matBk(1, 1);
        matBkout(4, 1) = matBk(2, 0);
        matBkout(4, 2) = matBk(2, 1);

        return matBkout;
    }
};

int main()
{
    EgoMotionModelAdapter egoMotionModelAdapter;

    return 0;
}
