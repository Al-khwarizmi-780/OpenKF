#ifndef OPENKF_RANGE_BEARING_MEAS_MODEL_H
#define OPENKF_RANGE_BEARING_MEAS_MODEL_H

#include "meas_model.h"
#include "types.h"

namespace kf
{
namespace measmodel
{
/// @brief Measurement space dimension for range-bearing measurement
/// model
/// \vec{z}=[pos_x, pos_y]^T
static constexpr int32_t DIM_Z_BEARING{2};

template <int32_t DIM_X>
class RangeBearingMeasModel
    : public MeasModel<RangeBearingMeasModel<DIM_X>, DIM_X, DIM_Z_BEARING>
{
 public:
  RangeBearingMeasModel(Vector<2> const& sensPos2D,
                        float32_t const rangeSigma = 1.0F,
                        float32_t const bearingSigma = 1.0F)
      : m_sensPos2D{sensPos2D}, m_rangeSigma{rangeSigma},
        m_bearingSigma{bearingSigma}
  {
  }
  ~RangeBearingMeasModel() {}

  static constexpr int32_t IDX_X_PX{0};  //< Index for position x in state
  static constexpr int32_t IDX_X_PY{1};  //< Index for position y in state

  static constexpr int32_t IDX_Z_RANGE{0};  //< Index for range in measurement
  static constexpr int32_t IDX_Z_BEARING{
      1};  //< Index for bearing in measurement

  /// @brief Measurement model function that maps the state space vector to
  /// the measurement space.
  /// @param vecX State space vector \vec{x}
  /// @return Measurement space vector \vec{z}
  Vector<DIM_Z_BEARING> h(Vector<DIM_X> const& vecX) const
  {
    Vector<DIM_Z_BEARING> vecZ;
    float32_t const px{vecX(IDX_X_PX) - m_sensPos2D(IDX_X_PX)};
    float32_t const py{vecX(IDX_X_PY) - m_sensPos2D(IDX_X_PY)};
    vecZ(IDX_Z_RANGE) = std::sqrt(px * px + py * py);
    vecZ(IDX_Z_BEARING) = std::atan2(py, px);
    return vecZ;
  }

  /// @brief Method that calculates the jacobians of the measurement model.
  /// @param vecX State Space vector \vec{x}
  /// @return The jacobians of the measurement model.
  Matrix<DIM_Z_BEARING, DIM_X> getJacobianHk(Vector<DIM_X> const& vecX) const
  {
    Matrix<DIM_Z_BEARING, DIM_X> matH;
    matH.setZero();

    float32_t const px{vecX(IDX_X_PX) - m_sensPos2D(IDX_X_PX)};
    float32_t const py{vecX(IDX_X_PY) - m_sensPos2D(IDX_X_PY)};
    float32_t const denom{px * px + py * py};

    matH(IDX_Z_RANGE, IDX_X_PX) = px / std::sqrt(denom);
    matH(IDX_Z_RANGE, IDX_X_PY) = py / std::sqrt(denom);
    matH(IDX_Z_BEARING, IDX_X_PX) = -py / denom;
    matH(IDX_Z_BEARING, IDX_X_PY) = px / denom;
    return matH;
  }

  /// @brief Get the measurement noise covariance R
  /// @return The measurement noise covariance R
  Matrix<DIM_Z_BEARING, DIM_Z_BEARING> getMeasurementNoiseCov() const
  {
    float32_t const rangeSigma2{m_rangeSigma * m_rangeSigma};
    float32_t const bearingSigma2{m_bearingSigma * m_bearingSigma};

    Matrix<DIM_Z_BEARING, DIM_Z_BEARING> matR;
    matR.setZero();
    matR(IDX_Z_RANGE, IDX_Z_RANGE) = rangeSigma2;
    matR(IDX_Z_BEARING, IDX_Z_BEARING) = bearingSigma2;
    return matR;
  }

  /// @brief Set range noise standard deviation
  /// @param rangeSigma Range measurement noise standard deviation
  void setRangeNoiseSigma(float32_t const rangeSigma)
  {
    m_rangeSigma = rangeSigma;
  }

  /// @brief Set bearing noise standard deviation
  /// @param bearingSigma Bearing measurement noise standard deviation
  void setBearingNoiseSigma(float32_t const bearingSigma)
  {
    m_bearingSigma = bearingSigma;
  }

 private:
  Vector<2> m_sensPos2D;     //< Sensor position in 2D
  float32_t m_rangeSigma;    //< Range measurement noise standard deviation
  float32_t m_bearingSigma;  //< Bearing measurement noise standard deviation
};
}  // namespace measmodel

}  // namespace kf
#endif  // OPENKF_RANGE_BEARING_MEAS_MODEL_H
