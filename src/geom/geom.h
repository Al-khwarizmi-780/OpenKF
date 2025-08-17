#ifndef OPENKF_LIB_GEOMETRY_H
#define OPENKF_LIB_GEOMETRY_H

#include "types.h"
#include <array>

namespace kf
{
    namespace geom
    {
        /// @brief 
        /// @param rect 
        /// @return 
        std::array<PointXY, 4> getRectangleCorners(Rectangle const& rect);

        /// @brief 
        /// @param point 
        /// @param rect 
        /// @return 
        bool checkPointInsideRectangle(PointXY const& point, Rectangle const& rect);
    }  // namespace geom
}  // namespace kf

#endif  // OPENKF_LIB_GEOMETRY_H
