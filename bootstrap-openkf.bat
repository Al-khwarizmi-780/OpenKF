@echo off

echo Creating Build Folder
md .\cpp\build

echo generating meta files
cmake -S .\cpp -B .\cpp\build

echo building ...
cmake --build .\cpp\build

echo installing ...
cmake --install .\cpp\build --config Debug
::runas /user:Administrator "cmake --install .\cpp\build --config Debug"

pause
