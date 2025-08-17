#include "geom.h"

std::array<kf::PointXY, 4> kf::geom::getRectangleCorners(
    kf::Rectangle const& rect)
{
    // Get four corners relative to rectangle axis coordinate
    double const hWidth{rect.dimension.width};
    double const hLength{rect.dimension.length};

    // Build the corner points relative to the rectangle center axis.
    kf::PointXY p1{-hWidth, hLength};   // top left
    kf::PointXY p2{hWidth, hLength};    // top right
    kf::PointXY p3{hWidth, -hLength};   // bottom right
    kf::PointXY p4{-hWidth, -hLength};  // bottom left

    // Temporarily variables to store calculated rotations
    double tmpX, tmpY;

    // Calculate cosine and sine
    double const cosTheta{std::cos(rect.angle)};
    double const sinTheta{std::sin(rect.angle)};

    // Rotate and translate P1 (top left)
    tmpX = p1.x * cosTheta - p1.y * sinTheta;
    tmpY = p1.x * sinTheta + p1.y * cosTheta;
    p1.x = rect.center.x + tmpX;
    p1.y = rect.center.y + tmpY;

    // Rotate and translate P2 (top right)
    tmpX = p2.x * cosTheta - p2.y * sinTheta;
    tmpY = p2.x * sinTheta + p2.y * cosTheta;
    p2.x = rect.center.x + tmpX;
    p2.y = rect.center.y + tmpY;

    // Rotate and translate P3 (bottom right)
    tmpX = p3.x * cosTheta - p3.y * sinTheta;
    tmpY = p3.x * sinTheta + p3.y * cosTheta;
    p3.x = rect.center.x + tmpX;
    p3.y = rect.center.y + tmpY;

    // Rotate and translate P4 (bottom left)
    tmpX = p4.x * cosTheta - p4.y * sinTheta;
    tmpY = p4.x * sinTheta + p4.y * cosTheta;
    p4.x = rect.center.x + tmpX;
    p4.y = rect.center.y + tmpY;

    // Fill results
    std::array<kf::PointXY, 4> const corners{p1, p2, p3, p4};

    return corners;
}

bool kf::geom::checkPointInsideRectangle(kf::PointXY const& point,
                                         kf::Rectangle const& rect)
{
    std::array<kf::PointXY, 4> const corners{getRectangleCorners(rect)};

    VecXY const AB{corners[1] - corners[0]};
    VecXY const AP{point - corners[0]};
    VecXY const BC{corners[2] - corners[1]};
    VecXY const BP{point - corners[1]};

    double const dotABAP{AB * AP};
    double const dotABAB{AB * AB};
    double const dotBCBP{BC * BP};
    double const dotBCBC{BC * BC};

    bool const isInside{(0 <= dotABAP) && (dotABAP <= dotABAB) &&
                        (0 <= dotBCBP) && (dotBCBP <= dotBCBC)};

    return isInside;
}