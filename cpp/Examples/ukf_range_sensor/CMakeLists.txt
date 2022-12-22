##
## Copyright 2022 CodingCorner
##
## Use of this source code is governed by an GPL-3.0 - style
## license that can be found in the LICENSE file or at
## https://opensource.org/licenses/GPL-3.0
##
## @author Mohanad Youssef <mohanad.magdy.hammad@gmail.com>
## @file CMakeLists.h
##

file(GLOB PROJECT_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

set(APPLICATION_NAME ${EXAMPLE_EXECUTABLE_PREFIX}_ukf_range_sensor)

add_executable(${APPLICATION_NAME} ${PROJECT_FILES})

set_target_properties(${APPLICATION_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${APPLICATION_NAME} PUBLIC Eigen3::Eigen)

target_include_directories(${APPLICATION_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../>
)
