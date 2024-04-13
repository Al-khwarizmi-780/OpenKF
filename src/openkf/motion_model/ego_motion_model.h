#ifndef OPENKF_EGO_MOTION_MODEL_H
#define OPENKF_EGO_MOTION_MODEL_H

#include "types.h"
#include "motion_model.h"

namespace kf
{
    namespace motionmodel
    {

        /// @brief State space dimension for ego-motion model \vec{x}=[x, y, yaw]^T
        static constexpr int32_t DIM_X{ 3 };

        /// @brief Input space dimension used by ego-motion model \vec{u}=[delta_distance, delta_yaw]^T
        static constexpr int32_t DIM_U{ 2 };

        /// @brief Derived class from base motion model class with external inputs
        /// Motion model for ego vehicle which is used for dead reckoning purposes
        /// by utilizing odometry external inputs vehicle displacement and change
        /// in heading angle and incrementing them to obtain the new state.
        class EgoMotionModel : public MotionModelExtInput<DIM_X, DIM_U>
        {
        public:
            
            EgoMotionModel()
                : m_qX(0.0F)
                , m_qY(0.0F)
                , m_qTheta(0.0F)
                , m_uDeltaDist(0.0F)
                , m_uDeltaYaw(0.0F)
            {}

            /// @brief Prediction motion model function that propagate the previous state to next state in time.
            /// @param vecX State space vector \vec{x}
            /// @param vecU Input space vector \vec{u}
            /// @param vecQ State white gaussian noise vector \vec{q}
            /// @return Predicted/ propagated state space vector
            virtual Vector<DIM_X> f(Vector<DIM_X> const & vecX, Vector<DIM_U> const & vecU, Vector<DIM_X> const & vecQ = Vector<DIM_X>::Zero()) override;

            /// @brief Get the process noise covariance Q
            /// @param vecX State space vector \vec{x}
            /// @param vecU Input space vector \vec{u}
            /// @return The process noise covariance Q
            virtual Matrix<DIM_X, DIM_X> getProcessNoiseCov(Vector<DIM_X> const & vecX, Vector<DIM_U> const & vecU) override;

            /// @brief Get the input noise covariance U
            /// @param vecX State space vector \vec{x}
            /// @param vecU Input space vector \vec{u}
            /// @return The input noise covariance U
            virtual Matrix<DIM_X, DIM_X> getInputNoiseCov(Vector<DIM_X> const & vecX, Vector<DIM_U> const & vecU) override;

            /// @brief Method that calculates the jacobians of the state transition model.
            /// @param vecX State Space vector \vec{x}
            /// @param vecU Input Space vector \vec{u}
            /// @return The jacobians of the state transition model.
            virtual Matrix<DIM_X, DIM_X> getJacobianFk(Vector<DIM_X> const & vecX, Vector<DIM_U> const & vecU) override;

            /// @brief Method that calculates the jacobians of the input transition model.
            /// @param vecX State Space vector \vec{x}
            /// @param vecU Input Space vector \vec{u}
            /// @return The jacobians of the input transition model.
            virtual Matrix<DIM_X, DIM_U> getJacobianBk(Vector<DIM_X> const & vecX, Vector<DIM_U> const & vecU) override;

            /// @brief Setter for noise variance of pose-x state.
            /// @param val variance value
            void setNoiseX(float32_t const val)
            {
                m_qX = val;
            }

            /// @brief Setter for noise variance of pose-y state.
            /// @param val variance value
            void setNoiseY(float32_t const val)
            {
                m_qY = val;
            }

            /// @brief Setter for noise variance of pose-yaw state.
            /// @param val variance value
            void setNoiseTheta(float32_t const val)
            {
                m_qTheta = val;
            }

            /// @brief Setter for noise variance of delta distance input.
            /// @param val variance value
            void setNoiseDeltaDist(float32_t const val)
            {
                m_uDeltaDist = val;
            }

            /// @brief Setter for noise variance of delta yaw input
            /// @param val variance value
            void setNoiseDeltaYaw(float32_t const val)
            {
                m_uDeltaYaw = val;
            }

        private:

            /// @brief The noise variance of pose-x state.
            float32_t m_qX;

            /// @brief The noise variance of pose-y state.
            float32_t m_qY;

            /// @brief The noise variance of pose-yaw state.
            float32_t m_qTheta;

            /// @brief The noise variance of delta distance input.
            float32_t m_uDeltaDist;

            /// @brief The noise variance of delta yaw input.
            float32_t m_uDeltaYaw;
        };
    }
}

#endif // OPENKF_EGO_MOTION_MODEL_H