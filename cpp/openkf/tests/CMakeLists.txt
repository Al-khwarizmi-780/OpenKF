if (BUILD_TESTING)
    cmake_minimum_required(VERSION 3.4)
    project(unit_tests)
  
    enable_testing()
    #find_package(GTest REQUIRED)
    #include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
        
    add_executable(${PROJECT_NAME} 
        unit_tests.cpp
        kalman_filter_test.cpp
        unscented_trasform_test.cpp
        unscented_kalman_filter_test.cpp
        square_root_ukf_test.cpp
    )

    target_link_libraries(${PROJECT_NAME} PUBLIC
        GTest::gtest_main
        OpenKF
    )

    if (MSVC)
        # https://stackoverflow.com/a/18635749
        add_compile_options(-MTd)
        set_property(TARGET ${PROJECT_NAME} PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif(MSVC)

    include(GoogleTest)
    gtest_discover_tests(${PROJECT_NAME})
    
endif(BUILD_TESTING)
