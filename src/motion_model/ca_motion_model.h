#include "motion_model.h"
#include "types.h"

namespace kf
{
namespace motionmodel
{
/// @brief State space dimension for constant acceleration motion model
/// \vec{x}=[pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]^T
static constexpr int32_t DIM_X_CA{6};

/// @brief Process noise dimension for constant acceleration motion model
static constexpr int32_t DIM_Q_CA{1};

class CaMotionModel : public MotionModel<CaMotionModel, DIM_X_CA>
{
 public:
  CaMotionModel(float32_t const sigma = 1.0F) { m_processNoiseVec[0] = sigma; }
  ~CaMotionModel() {}

  /// @brief Prediction motion model function that propagate the previous state
  /// to next state in time.
  /// @param vecX State space vector \vec{x}
  /// @param dt Time step between state updates (unit: seconds)
  /// @return Predicted/ propagated state space vector
  Vector<DIM_X_CA> f(Vector<DIM_X_CA> const& vecX, float32_t dt = 1.0F) const;

  /// @brief Get the process noise covariance Q
  /// @param vecX State space vector \vec{x}
  /// @param dt Time step between state updates (unit: seconds)
  /// @return The process noise covariance Q
  Matrix<DIM_X_CA, DIM_X_CA> getProcessNoiseCov(Vector<DIM_X_CA> const& vecX,
                                                float32_t dt = 1.0F) const;

  /// @brief Method that calculates the jacobians of the state transition model.
  /// @param vecX State Space vector \vec{x}
  /// @param dt Time step between state updates (unit: seconds)
  /// @return The jacobians of the state transition model.
  Matrix<DIM_X_CA, DIM_X_CA> getJacobianFk(Vector<DIM_X_CA> const& vecX,
                                           float32_t dt = 1.0F) const;

  /// @brief Get state dimension
  /// @return State dimension
  static constexpr int32_t getStateDim() { return DIM_X_CA; }

  /// @brief Set process noise standard deviation
  /// @param sigma Process noise standard deviation
  void setSigma(float32_t const sigma) { m_processNoiseVec[0] = sigma; }

  static constexpr int32_t IDX_PX{0};  //< Index for position x
  static constexpr int32_t IDX_PY{1};  //< Index for position y
  static constexpr int32_t IDX_VX{2};  //< Index for velocity x
  static constexpr int32_t IDX_VY{3};  //< Index for velocity y
  static constexpr int32_t IDX_AX{4};  //< Index for acceleration x
  static constexpr int32_t IDX_AY{5};  //< Index for acceleration y

 private:
  Vector<DIM_Q_CA> m_processNoiseVec;  //< Process noise vector
};

}  // namespace motionmodel
}  // namespace kf