#include "ego_motion_model.h"

namespace kf
{
    namespace motionmodel
    {
        Vector<DIM_X> EgoMotionModel::f(Vector<DIM_X> const & vecX, Vector<DIM_U> const & vecU, Vector<DIM_X> const & vecQ) const
        {
            Vector<DIM_X> vecXout;
            float32_t & oPosX{ vecXout[0] };
            float32_t & oPosY{ vecXout[1] };
            float32_t & oYaw { vecXout[2] };

            float32_t const & iPosX{ vecX[0] };
            float32_t const & iPosY{ vecX[1] };
            float32_t const & iYaw { vecX[2] };

            float32_t const & qPosX{ vecQ[0] };
            float32_t const & qPosY{ vecQ[1] };
            float32_t const & qYaw { vecQ[2] };

            float32_t const & deltaDist{ vecU[0] };
            float32_t const & deltaYaw { vecU[1] };

            float32_t const halfDeltaYaw{ deltaYaw / 2.0F };
            float32_t const iYawPlusHalfDeltaYaw{ iYaw + halfDeltaYaw };

            oPosX = iPosX + (deltaDist * std::cos(iYawPlusHalfDeltaYaw)) + qPosX;
            oPosY = iPosY + (deltaDist * std::sin(iYawPlusHalfDeltaYaw)) + qPosY;
            oYaw  = iYaw  + deltaYaw + qYaw;

            return vecXout;
        }

        Matrix<DIM_X,DIM_X> EgoMotionModel::getProcessNoiseCov(Vector<DIM_X> const & vecX, Vector<DIM_U> const & vecU) const
        {
            Matrix<DIM_X,DIM_X> matQk;

            float32_t & q11 = matQk(0, 0);
            float32_t & q12 = matQk(0, 1);
            float32_t & q13 = matQk(0, 2);

            float32_t & q21 = matQk(1, 0);
            float32_t & q22 = matQk(1, 1);
            float32_t & q23 = matQk(1, 2);

            float32_t & q31 = matQk(2, 0);
            float32_t & q32 = matQk(2, 1);
            float32_t & q33 = matQk(2, 2);

            float32_t const & iYaw { vecX[2] };

            float32_t const & deltaDist{ vecU[0] };
            float32_t const & deltaYaw { vecU[1] };

            float32_t const deltaDistSquare{ deltaDist * deltaDist };
            float32_t const halfDeltaYaw{ deltaYaw / 2.0F };
            float32_t const iYawPlusHalfDeltaYaw{ iYaw + halfDeltaYaw };
            float32_t const sinTheta{ std::sin(iYawPlusHalfDeltaYaw) };
            float32_t const cosTheta{ std::cos(iYawPlusHalfDeltaYaw) };
            float32_t const sinCosTheta{ sinTheta * cosTheta };

            q11 = m_qX + (deltaDistSquare * sinTheta * sinTheta * m_qTheta);
            q12 = -deltaDistSquare * sinCosTheta * m_qTheta;
            q13 = -deltaDist * sinTheta * m_qTheta;

            q21 = -deltaDistSquare * sinCosTheta * m_qTheta;
            q22 = m_qY + (deltaDistSquare * cosTheta * cosTheta * m_qTheta);
            q23 = deltaDist * cosTheta * m_qTheta;

            q31 = -deltaDist * sinTheta * m_qTheta;
            q32 = deltaDist * cosTheta * m_qTheta;
            q33 = m_qTheta;

            return matQk;
        }

        Matrix<DIM_X, DIM_X> EgoMotionModel::getInputNoiseCov(Vector<DIM_X> const &vecX, Vector<DIM_U> const &vecU) const
        {
            Matrix<DIM_X,DIM_X> matUk;

            float32_t & u11 = matUk(0, 0);
            float32_t & u12 = matUk(0, 1);
            float32_t & u13 = matUk(0, 2);

            float32_t & u21 = matUk(1, 0);
            float32_t & u22 = matUk(1, 1);
            float32_t & u23 = matUk(1, 2);

            float32_t & u31 = matUk(2, 0);
            float32_t & u32 = matUk(2, 1);
            float32_t & u33 = matUk(2, 2);

            float32_t const & iYaw { vecX[2] };

            float32_t const & deltaDist{ vecU[0] };
            float32_t const & deltaYaw { vecU[1] };

            float32_t const deltaDistSquare{ deltaDist * deltaDist };
            float32_t const halfDeltaDistSquare{ deltaDist / 2.0F };
            float32_t const quarterDeltaDistSquare{ deltaDistSquare / 4.0F };
            float32_t const halfDeltaYaw{ deltaYaw / 2.0F };
            float32_t const iYawPlusHalfDeltaYaw{ iYaw + halfDeltaYaw };
            float32_t const sinTheta{ std::sin(iYawPlusHalfDeltaYaw) };
            float32_t const cosTheta{ std::cos(iYawPlusHalfDeltaYaw) };
            float32_t const sin2Theta{ sinTheta * sinTheta };
            float32_t const cos2Theta{ cosTheta * cosTheta };
            float32_t const sinCosTheta{ sinTheta * cosTheta };

            u11 = (m_uDeltaDist * cos2Theta) + (m_uDeltaYaw * quarterDeltaDistSquare * sin2Theta);
            u12 = (m_uDeltaDist * sinCosTheta) - (m_uDeltaYaw * quarterDeltaDistSquare * sinCosTheta);
            u13 = -m_uDeltaYaw * halfDeltaDistSquare * sinTheta;

            u21 = (m_uDeltaDist * sinCosTheta) - (m_uDeltaYaw * quarterDeltaDistSquare * sinCosTheta);
            u22 = (m_uDeltaDist * sin2Theta) + (m_uDeltaYaw * quarterDeltaDistSquare * cos2Theta);
            u23 = m_uDeltaYaw * halfDeltaDistSquare * cosTheta;

            u31 = -m_uDeltaYaw * halfDeltaDistSquare * sinTheta;
            u32 = m_uDeltaYaw * halfDeltaDistSquare * cosTheta;
            u33 = m_uDeltaYaw;

            return matUk;
        }

        Matrix<DIM_X,DIM_X> EgoMotionModel::getJacobianFk(Vector<DIM_X> const & vecX, Vector<DIM_U> const & vecU) const
        {
            Matrix<DIM_X,DIM_X> matFk;

            float32_t & df1dx1 = matFk(0, 0);
            float32_t & df1dx2 = matFk(0, 1);
            float32_t & df1dx3 = matFk(0, 2);

            float32_t & df2dx1 = matFk(1, 0);
            float32_t & df2dx2 = matFk(1, 1);
            float32_t & df2dx3 = matFk(1, 2);

            float32_t & df3dx1 = matFk(2, 0);
            float32_t & df3dx2 = matFk(2, 1);
            float32_t & df3dx3 = matFk(2, 2);

            float32_t const & iYaw { vecX[2] };

            float32_t const & deltaDist{ vecU[0] };
            float32_t const & deltaYaw { vecU[1] };

            float32_t const halfDeltaYaw{ deltaYaw / 2.0F };
            float32_t const iYawPlusHalfDeltaYaw{ iYaw + halfDeltaYaw };

            df1dx1 = 1.0F;
            df1dx2 = 0.0F;
            df1dx3 = -deltaDist * std::sin(iYawPlusHalfDeltaYaw);

            df2dx1 = 0.0F;
            df2dx2 = 1.0F;
            df2dx3 = deltaDist * std::cos(iYawPlusHalfDeltaYaw);

            df3dx1 = 0.0F;
            df3dx2 = 0.0F;
            df3dx3 = 1.0F;

            return matFk;
        }

        Matrix<DIM_X,DIM_U> EgoMotionModel::getJacobianBk(Vector<DIM_X> const & vecX, Vector<DIM_U> const & vecU) const
        {
            Matrix<DIM_X,DIM_U> matBk;

            float32_t & df1du1 = matBk(0, 0);
            float32_t & df1du2 = matBk(0, 1);

            float32_t & df2du1 = matBk(1, 0);
            float32_t & df2du2 = matBk(1, 1);

            float32_t & df3du1 = matBk(2, 0);
            float32_t & df3du2 = matBk(2, 1);

            float32_t const & iYaw { vecX[2] };

            float32_t const & deltaDist{ vecU[0] };
            float32_t const & deltaYaw { vecU[1] };

            float32_t const halfDeltaDist{ deltaDist / 2.0F };
            float32_t const halfDeltaYaw { deltaYaw / 2.0F };

            float32_t const iYawPlusHalfDeltaYaw{ iYaw + halfDeltaYaw };

            df1du1 = std::cos(iYawPlusHalfDeltaYaw);
            df1du2 = -halfDeltaDist * std::sin(iYawPlusHalfDeltaYaw);

            df2du1 = std::sin(iYawPlusHalfDeltaYaw);
            df2du2 = halfDeltaDist * std::cos(iYawPlusHalfDeltaYaw);

            df3du1 = 0.0F;
            df3du2 = 1.0F;

            return matBk;
        }
    }
}