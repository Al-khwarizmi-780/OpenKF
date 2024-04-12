cmake_minimum_required(VERSION 3.4)

# ============================================================================================
# VCPKG Toolchain
# ============================================================================================
if(WIN32)
    # use vcpkg as packages manager in windows platform
    # environment variable needs to be added for the path to vcpkg installation "VCPKG_ROOT"
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
endif(WIN32)

# ============================================================================================
# ============================================================================================
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/build/package)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

add_subdirectory(src/third_party/googletest)
add_subdirectory(src/openkf)
add_subdirectory(src/examples)
add_subdirectory(tests)
