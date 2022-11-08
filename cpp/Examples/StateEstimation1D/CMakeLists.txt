##
## @author Mohanad Youssef
## @file KalmanFilterExercise/Examples/StateEstimate1D/CMakeLists.txt
##

file(GLOB PROJECT_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

set(APPLICATION_NAME ${CMAKE_PROJECT_NAME}_estimate_1D)

add_executable(${APPLICATION_NAME} ${PROJECT_FILES})

set_target_properties(${APPLICATION_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${APPLICATION_NAME} PUBLIC Eigen3::Eigen)

target_include_directories(${APPLICATION_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../>
)
