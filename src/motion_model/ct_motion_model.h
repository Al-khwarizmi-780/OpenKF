#include "motion_model.h"
#include "types.h"

namespace kf
{
namespace motionmodel
{
/// @brief State space dimension for constant turn rate motion model
/// \vec{x}=[pos_x, pos_y, v, theta, omega]^T
static constexpr int32_t DIM_X_CT{5};

/// @brief Process noise dimension for constant turn rate motion model
static constexpr int32_t DIM_Q_CT{2};

class CtMotionModel : public MotionModel<CtMotionModel, DIM_X_CT>
{
 public:
  CtMotionModel(float32_t const sigmaV = 1.0F,
                float32_t const sigmaOmega = 1.0F)
  {
    m_processNoiseVec[0] = sigmaV;
    m_processNoiseVec[1] = sigmaOmega;
  }
  ~CtMotionModel() {}

  /// @brief Prediction motion model function that propagate the previous state
  /// to next state in time.
  /// @param vecX State space vector \vec{x}
  /// @param dt Time step between state updates (unit: seconds)
  /// @return Predicted/ propagated state space vector
  Vector<DIM_X_CT> f(Vector<DIM_X_CT> const& vecX, float32_t dt = 1.0F) const;

  /// @brief Get the process noise covariance Q
  /// @param vecX State space vector \vec{x}
  /// @param dt Time step between state updates (unit: seconds)
  /// @return The process noise covariance Q
  Matrix<DIM_X_CT, DIM_X_CT> getProcessNoiseCov(Vector<DIM_X_CT> const& vecX,
                                                float32_t dt = 1.0F) const;

  /// @brief Method that calculates the jacobians of the state transition model.
  /// @param vecX State Space vector \vec{x}
  /// @param dt Time step between state updates (unit: seconds)
  /// @return The jacobians of the state transition model.
  Matrix<DIM_X_CT, DIM_X_CT> getJacobianFk(Vector<DIM_X_CT> const& vecX,
                                           float32_t dt = 1.0F) const;

  /// @brief Get state dimension
  /// @return State dimension
  static constexpr int32_t getStateDim() { return DIM_X_CT; }

  /// @brief Set velocity process noise standard deviation.
  /// @param sigmaV Velocity process noise standard deviation
  void setSigmaV(float32_t const sigmaV) { m_processNoiseVec[0] = sigmaV; }

  /// @brief Set turn rate process noise standard deviation.
  /// @param sigmaOmega Turn rate process noise standard deviation
  void setSigmaOmega(float32_t const sigmaOmega)
  {
    m_processNoiseVec[1] = sigmaOmega;
  }

  static constexpr int32_t IDX_PX{0};     //< Index for position x
  static constexpr int32_t IDX_PY{1};     //< Index for position y
  static constexpr int32_t IDX_V{2};      //< Index for velocity
  static constexpr int32_t IDX_THETA{3};  //< Index for heading angle
  static constexpr int32_t IDX_OMEGA{4};  //< Index for yaw rate

 private:
  Vector<DIM_Q_CT> m_processNoiseVec;  //< Process noise vector
};

}  // namespace motionmodel
}  // namespace kf