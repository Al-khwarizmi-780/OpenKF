cmake_minimum_required(VERSION 3.4)

# ============================================================================================
# ============================================================================================
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/build/package)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_GTEST ON CACHE BOOL "" FORCE)

project(OpenKF)

set(INCLUDE_FOLDER "include")
set(LIBRARY_INSTALL_DIR "lib")
set(INCLUDE_INSTALL_DIR "${INCLUDE_FOLDER}/${PROJECT_NAME}")
set(CONFIG_INSTALL_DIR  "${LIBRARY_INSTALL_DIR}/cmake/${PROJECT_NAME}")
set(namespace "%{PROJECT_NAME}::")
set(TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")

enable_language(C CXX)

if (NOT MSVC)
    set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra")
endif(NOT MSVC)

if (MSVC)
    # https://stackoverflow.com/a/18635749
    add_compile_options(-MTd)
endif (MSVC)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include(CTest)

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tests)
