# OpenKF (The Kalman Filter Library)

This is an open source C++ Kalman filter library based on Eigen3 library for matrix operations.

The library has generic template based classes for most of Kalman filter variants including:

1. Kalman Filter
2. Extended Kalman Filter
3. Unscented Kalman Filter
4. Square-root Unscented Kalman Filter

**LICENSE**: [GPL-3.0 license](LICENSE.md)

**Author**: Mohanad Youssef ([codingcorner.org](https://codingcorner.org/))

**YouTube Channel**: [https://www.youtube.com/@al-khwarizmi](https://www.youtube.com/@al-khwarizmi) 

![](.res/images/codingcorner_cover_image.png)

## Getting Started

One can build the library and install the files in the system to be used in different external projects.

You just need to execute the batch file ``bootstrap-openkf.bat`` from a PowerShell Terminal (in Administrator Mode).

```batch
>> ./bootstrap-openkf.bat
```

This batch file will execute cmake commands to generate meta files, build, and install the library files in the system.

After that, the OpenKF library is ready to be used in external project.

In the **_CMakeLists.txt_** you must include these three lines of code:

````cmake
find_package(OpenKF REQUIRED)
target_link_libraries(<your-project-name> PUBLIC OpenKF)
target_include_directories(<your-project-name> PUBLIC ${OPENKF_INCLUDE_DIR})
````
