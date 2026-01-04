#include "motion_model.h"
#include "types.h"

namespace kf
{
namespace motionmodel
{
/// @brief State space dimension for constant turn rate motion model
/// \vec{x}=[pos_x, pos_y, v, theta, omega]^T
static constexpr int32_t DIM_X_CT{5};

class CtMotionModel : public MotionModel<CtMotionModel, DIM_X_CT>
{
 public:
  CtMotionModel() {}
  ~CtMotionModel() {}

  /// @brief Prediction motion model function that propagate the previous state
  /// to next state in time.
  /// @param vecX State space vector \vec{x}
  /// @param dt Time step between state updates (unit: seconds)
  /// @return Predicted/ propagated state space vector
  Vector<DIM_X_CT> f(Vector<DIM_X_CT> const& vecX, float32_t dt = 1.0F) const;

  /// @brief Get the process noise covariance Q
  /// @param sigma Standard deviation of the process noise v=[sigma_a,
  /// sigma_alpha]
  /// @param dt Time step between state updates (unit: seconds)
  /// @return The process noise covariance Q
  template <int32_t DIM_SIGMA>
  Matrix<DIM_X_CT, DIM_X_CT> getProcessNoiseCov(Vector<DIM_SIGMA> const& sigma,
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

 private:
  static constexpr int32_t IDX_PX{0};     //< Index for position x
  static constexpr int32_t IDX_PY{1};     //< Index for position y
  static constexpr int32_t IDX_V{2};      //< Index for velocity
  static constexpr int32_t IDX_THETA{3};  //< Index for heading angle
  static constexpr int32_t IDX_OMEGA{4};  //< Index for yaw rate
};

}  // namespace motionmodel
}  // namespace kf