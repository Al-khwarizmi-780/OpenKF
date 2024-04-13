# cmake_minimum_required(VERSION 3.4)
set(TEST_APP_NAME ${PROJECT_NAME}_Test)

enable_testing()

# find_package(GTest REQUIRED)
# include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
    
add_executable(${TEST_APP_NAME} 
    unit_tests.cpp
    kalman_filter/kalman_filter_test.cpp
    kalman_filter/unscented_trasform_test.cpp
    kalman_filter/unscented_kalman_filter_test.cpp
    kalman_filter/square_root_ukf_test.cpp
    motion_model/ego_motion_model_test.cpp
)

target_link_libraries(${TEST_APP_NAME} PRIVATE
    GTest::gtest_main
    GTest::gmock
    OpenKF
)

target_include_directories( ${TEST_APP_NAME} PRIVATE ${GTEST_INCLUDE_DIRS} )
# target_link_libraries( ${TEST_APP_NAME} tested_library ${GTEST_BOTH_LIBRARIES} )

if (MSVC)
    # https://stackoverflow.com/a/18635749
    set_property(TARGET ${TEST_APP_NAME} PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif(MSVC)

include(GoogleTest)
gtest_discover_tests(${TEST_APP_NAME})
