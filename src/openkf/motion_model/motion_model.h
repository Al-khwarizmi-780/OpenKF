#ifndef OPENKF_MOTION_MODEL_H
#define OPENKF_MOTION_MODEL_H

#include "types.h"

namespace kf
{
    /// @brief Base class for motion models used by kalman filters
    /// @tparam DIM_X State space vector dimension
    template<int32_t DIM_X>
    class MotionModel
    {
    public:

        /// @brief Prediction motion model function that propagate the previous state to next state in time.
        /// @param vecX State space vector \vec{x}
        /// @param vecQ State white gaussian noise vector \vec{q}
        /// @return Predicted/ propagated state space vector
        virtual Vector<DIM_X> f(Vector<DIM_X> const & vecX, Vector<DIM_X> const & vecQ = Vector<DIM_X>::Zero()) const = 0;

        /// @brief Get the process noise covariance Q
        /// @param vecX State space vector \vec{x}
        /// @return The process noise covariance Q
        virtual Matrix<DIM_X, DIM_X> getProcessNoiseCov(Vector<DIM_X> const & vecX) const = 0; 

        /// @brief Method that calculates the jacobians of the state transition model.
        /// @param vecX State Space vector \vec{x}
        /// @return The jacobians of the state transition model.
        virtual Matrix<DIM_X, DIM_X> getJacobianFk(Vector<DIM_X> const & vecX) const = 0;
    };

    /// @brief Base class for motion models with external inputs used by kalman filters
    /// @tparam DIM_X State space vector dimension
    /// @tparam DIM_U Input space vector dimension
    template<int32_t DIM_X, int32_t DIM_U>
    class MotionModelExtInput
    {
    public:

        /// @brief Prediction motion model function that propagate the previous state to next state in time.
        /// @param vecX State space vector \vec{x}
        /// @param vecU Input space vector \vec{u}
        /// @param vecQ State white gaussian noise vector \vec{q}
        /// @return Predicted/ propagated state space vector
        virtual Vector<DIM_X> f(Vector<DIM_X> const & vecX, Vector<DIM_U> const & vecU, Vector<DIM_X> const & vecQ = Vector<DIM_X>::Zero()) const = 0;

        /// @brief Get the process noise covariance Q
        /// @param vecX State space vector \vec{x}
        /// @param vecU Input space vector \vec{u}
        /// @return The process noise covariance Q
        virtual Matrix<DIM_X, DIM_X> getProcessNoiseCov(Vector<DIM_X> const & vecX, Vector<DIM_U> const & vecU) const = 0; 

        /// @brief Get the input noise covariance U
        /// @param vecX State space vector \vec{x}
        /// @param vecU Input space vector \vec{u}
        /// @return The input noise covariance U
        virtual Matrix<DIM_X, DIM_X> getInputNoiseCov(Vector<DIM_X> const & vecX, Vector<DIM_U> const & vecU) const = 0; 

        /// @brief Method that calculates the jacobians of the state transition model.
        /// @param vecX State Space vector \vec{x}
        /// @param vecU Input Space vector \vec{u}
        /// @return The jacobians of the state transition model.
        virtual Matrix<DIM_X, DIM_X> getJacobianFk(Vector<DIM_X> const & vecX, Vector<DIM_U> const & vecU) const = 0;

        /// @brief Method that calculates the jacobians of the input transition model.
        /// @param vecX State Space vector \vec{x}
        /// @param vecU Input Space vector \vec{u}
        /// @return The jacobians of the input transition model.
        virtual Matrix<DIM_X, DIM_U> getJacobianBk(Vector<DIM_X> const & vecX, Vector<DIM_U> const & vecU) const = 0;
    };
}

#endif // OPENKF_MOTION_MODEL_H
