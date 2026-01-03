#include "motion_model.h"
#include "types.h"

namespace kf
{
namespace motionmodel
{
/// @brief State space dimension for constant velocity motion model
/// \vec{x}=[pos_x, pos_y, vel_x, vel_y]^T
static constexpr int32_t DIM_X_CV{4};

class CvMotionModel : public MotionModel<CvMotionModel, DIM_X_CV>
{
 public:
  CvMotionModel() {}
  ~CvMotionModel() {}

  /// @brief Prediction motion model function that propagate the previous state
  /// to next state in time.
  /// @param vecX State space vector \vec{x}
  /// @param dt Time step between state updates (unit: seconds)
  /// @return Predicted/ propagated state space vector
  Vector<DIM_X_CV> f(Vector<DIM_X_CV> const& vecX, float32_t dt = 1.0F) const;

  /// @brief Get the process noise covariance Q
  /// @param sigma Standard deviation of the process noise
  /// @param dt Time step between state updates (unit: seconds)
  /// @return The process noise covariance Q
  Matrix<DIM_X_CV, DIM_X_CV> getProcessNoiseCov(float32_t sigma,
                                                float32_t dt = 1.0F) const;

  /// @brief Method that calculates the jacobians of the state transition model.
  /// @param vecX State Space vector \vec{x}
  /// @param dt Time step between state updates (unit: seconds)
  /// @return The jacobians of the state transition model.
  Matrix<DIM_X_CV, DIM_X_CV> getJacobianFk(Vector<DIM_X_CV> const& vecX,
                                           float32_t dt = 1.0F) const;

  /// @brief Get state dimension
  /// @return State dimension
  static constexpr int32_t getStateDim() { return DIM_X_CV; }

 private:
  static constexpr int32_t IDX_PX{0};  //< Index for position x
  static constexpr int32_t IDX_PY{1};  //< Index for position y
  static constexpr int32_t IDX_VX{2};  //< Index for velocity x
  static constexpr int32_t IDX_VY{3};  //< Index for velocity y
};

}  // namespace motionmodel
}  // namespace kf