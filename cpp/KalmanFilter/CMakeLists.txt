##
## @author Mohanad Youssef
## @file KalmanFilterExercise/KalmanFilter/CMakeLists.txt
##

file(GLOB LIBRARY_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

set(LIBRARY_NAME ${CMAKE_PROJECT_NAME}_lib)

add_library(${LIBRARY_NAME} ${LIBRARY_FILES})

set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${LIBRARY_NAME} PUBLIC Eigen3::Eigen)

target_include_directories(${LIBRARY_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)