#include "motion_model.h"
#include "types.h"

namespace kf
{
namespace motionmodel
{
/// @brief State space dimension for constant velocity motion model
/// \vec{x}=[pos_x, pos_y, vel_x, vel_y]^T
static constexpr int32_t DIM_X_CV{4};

/// @brief Process noise dimension for constant velocity motion model
static constexpr int32_t DIM_Q_CV{1};

class CvMotionModel : public MotionModel<CvMotionModel, DIM_X_CV>
{
 public:
  CvMotionModel(float32_t const processNoise = 1.0F)
  {
    m_processNoiseVec[0] = processNoise;
  }
  ~CvMotionModel() {}

  /// @brief Prediction motion model function that propagate the previous state
  /// to next state in time.
  /// @param vecX State space vector \vec{x}
  /// @param dt Time step between state updates (unit: seconds)
  /// @return Predicted/ propagated state space vector
  Vector<DIM_X_CV> f(Vector<DIM_X_CV> const& vecX, float32_t dt = 1.0F) const;

  /// @brief Get the process noise covariance Q
  /// @param vecX State space vector \vec{x}
  /// @param dt Time step between state updates (unit: seconds)
  /// @return The process noise covariance Q
  Matrix<DIM_X_CV, DIM_X_CV> getProcessNoiseCov(Vector<DIM_X_CV> const& vecX,
                                                float32_t dt = 1.0F) const;

  /// @brief Method that calculates the jacobians of the state transition model.
  /// @param vecX State Space vector \vec{x}
  /// @param dt Time step between state updates (unit: seconds)
  /// @return The jacobians of the state transition model.
  Matrix<DIM_X_CV, DIM_X_CV> getJacobianFk(Vector<DIM_X_CV> const& vecX,
                                           float32_t dt = 1.0F) const;

  /// @brief Set process noise standard deviation
  /// @param sigma Process noise standard deviation
  void setSigma(float32_t const sigma) { m_processNoiseVec[0] = sigma; }

  static constexpr int32_t IDX_PX{0};  //< Index for position x
  static constexpr int32_t IDX_PY{1};  //< Index for position y
  static constexpr int32_t IDX_VX{2};  //< Index for velocity x
  static constexpr int32_t IDX_VY{3};  //< Index for velocity y

 private:
  Vector<DIM_Q_CV> m_processNoiseVec;  //< Process noise vector
};

}  // namespace motionmodel
}  // namespace kf