##
## Copyright 2022 Mohanad Youssef (Al-khwarizmi)
##
## Use of this source code is governed by an GPL-3.0 - style
## license that can be found in the LICENSE file or at
## https://opensource.org/licenses/GPL-3.0
##
## @author Mohanad Youssef <mohanad.magdy.hammad@gmail.com>
## @file CMakeLists.h
##

# file(GLOB LIBRARY_FILES
#     "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
#     "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
#     )

set(LIBRARY_SRC_FILES
    dummy.cpp
)

set(LIBRARY_HDR_FILES
    types.h
    util.h
    kalman_filter.h
    unscented_transform.h
    unscented_kalman_filter.h
    square_root_ukf.h
)

set(LIBRARY_NAME ${PROJECT_NAME})

add_library(${LIBRARY_NAME} ${LIBRARY_SRC_FILES} ${LIBRARY_HDR_FILES})

set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${LIBRARY_NAME} PUBLIC Eigen3::Eigen)

target_include_directories(${LIBRARY_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${INCLUDE_INSTALL_DIR}>
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/conf/Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${LIBRARY_NAME}Config.cmake
    INSTALL_DESTINATION ${CONFIG_INSTALL_DIR}
    PATH_VARS INCLUDE_FOLDER
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${LIBRARY_NAME}ConfigVersion.cmake
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion
)

install(
    FILES ${LIBRARY_HDR_FILES} 
    DESTINATION ${INCLUDE_INSTALL_DIR}
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${LIBRARY_NAME}Config.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/${LIBRARY_NAME}ConfigVersion.cmake
    DESTINATION ${CONFIG_INSTALL_DIR}
)

install(
    TARGETS ${LIBRARY_NAME} 
    EXPORT "${TARGETS_EXPORT_NAME}"
    LIBRARY DESTINATION "${LIBRARY_INSTALL_DIR}"
    ARCHIVE DESTINATION "${LIBRARY_INSTALL_DIR}"
    INCLUDES DESTINATION "${INCLUDE_FOLDER}"
)

# Config
#   * <prefix>/lib/cmake/OpenKF/OpenKFTargets.cmake
install(
    EXPORT ${TARGETS_EXPORT_NAME} 
    DESTINATION ${CONFIG_INSTALL_DIR}
)