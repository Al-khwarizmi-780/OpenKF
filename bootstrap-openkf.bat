@echo off

    if not exist
    ".\cpp\build"(echo Creating./ cpp / build Folder md./ cpp /
                  build) else(echo./ cpp / build folder already exists)

        echo generating meta files cmake -
    S./ cpp -
    B./ cpp /
        build

            echo building... cmake-- build./
        cpp /
        build

            echo installing... cmake-- install./
        cpp / build-- config Debug ::runas /
        user : Administrator "cmake --install .\cpp\build --config Debug"

               pause
